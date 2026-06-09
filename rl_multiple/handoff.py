"""
Reconstruct a `SimulationEnv` to mid-scenario state captured in a
HumanDataRecorder CSV.

The CSV captures one row per aircraft per simulated second. A "frame"
is the set of rows sharing the same `sim_time`. To freeze a scenario at
frame N and resume from there with a different policy, we:

1. Wipe `sim.aircraft_list` and reset counters.
2. For each row in the frame, construct an `Aircraft` with the recorded
   pose (x_nm/y_nm → pixels), targets, STAR remainder, and approach
   flags (`ils_runway`, `loc`, `gs`, `on_ground`).
3. Reset `sim.sim_time` to the frame's timestamp.
4. Freeze spawning (the resumed rollout is meant to be A/B-able — no
   new arrivals to confound the comparison).

Used by both the CLI tool (`rl_multiple.rollout_from_frame`) and the
watch handoff button (`/replay/handoff`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from environment.core.aircraft import Aircraft
from environment.core.simulation import SimulationEnv


def _float_or(v, default=0.0):
    try:
        return float(v) if v not in (None, '') else default
    except (TypeError, ValueError):
        return default


def _bool_from_csv(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ('true', '1', 't', 'yes')


def load_frames_from_csv(csv_path: str | Path) -> list[dict]:
    """Group CSV rows by `sim_time` into a list of frame dicts, each:

        {'sim_time': float, 'rows': [row_dict, ...]}

    Frames are sorted by sim_time. Each `row_dict` is the raw CSV row
    (all values as strings) — converted to typed fields later.
    """
    by_time: dict[float, list[dict]] = {}
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = _float_or(row.get('sim_time'))
            by_time.setdefault(t, []).append(row)
    return [{'sim_time': t, 'rows': by_time[t]} for t in sorted(by_time.keys())]


def reconstruct_from_frame(sim: SimulationEnv, frame: dict) -> int:
    """Wipe `sim` of any current aircraft and rebuild from `frame['rows']`.

    `frame` is one element of `load_frames_from_csv(...)`. Spawning is
    NOT touched here — callers should set `sim.spawn_rate` to disable
    it (or monkey-patch `sim.spawner.update`) before stepping.

    Returns the number of aircraft reconstructed.
    """
    sim.aircraft_list.clear()
    sim.sim_time = float(frame.get('sim_time', 0.0))
    sim.crash_occurred = False
    sim.crash_message = ''
    sim.has_violation = False

    nmpp = sim.nm_per_pixel
    ax = sim.airport_x
    ay = sim.airport_y
    procedures = sim.data.get('star_procedures', {})

    n = 0
    for row in frame.get('rows', []):
        cs = (row.get('callsign') or '').strip().upper()
        if not cs:
            continue
        # Skip rows already marked terminal in the CSV — they belong to
        # the last tick of an aircraft's life, not a state to resume.
        if (row.get('terminal') or '').strip():
            continue

        x_nm = _float_or(row.get('x_nm'))
        y_nm = _float_or(row.get('y_nm'))
        px = x_nm / nmpp + ax
        py = ay - y_nm / nmpp

        hdg = _float_or(row.get('heading'))
        alt = _float_or(row.get('altitude'))
        spd = _float_or(row.get('airspeed'), default=250.0)

        ac = Aircraft(cs, px, py, hdg, alt, spd, nmpp, sim.coords)
        ac.target_heading = _float_or(row.get('target_heading'), default=hdg)
        ac.target_altitude = _float_or(row.get('target_altitude'), default=alt)
        ac.target_airspeed = _float_or(row.get('target_airspeed'), default=spd)

        # STAR — trim the procedure to start at the row's current
        # target_wpt so the aircraft doesn't backtrack to step 0.
        star_name = (row.get('star') or '').strip() or None
        target_wpt = (row.get('target_wpt') or '').strip() or None
        if star_name and star_name in procedures:
            steps = procedures[star_name]
            if target_wpt:
                for i, s in enumerate(steps):
                    if s.get('waypoint') == target_wpt:
                        steps = steps[i:]
                        break
            ac.assign_star(steps, name=star_name)
            # assign_star overwrites target_heading via _apply_star_head's
            # waypoint-bearing calc only when target_wpt changes; preserve
            # the CSV's recorded targets to keep continuity.
            ac.target_heading = _float_or(row.get('target_heading'),
                                          default=ac.target_heading)
            ac.target_altitude = _float_or(row.get('target_altitude'),
                                           default=ac.target_altitude)
            ac.target_airspeed = _float_or(row.get('target_airspeed'),
                                           default=ac.target_airspeed)
        elif target_wpt:
            # No STAR, but heading-to-waypoint navigation.
            ac.target_wpt = target_wpt

        # Approach state. Prefer the explicit `ils_runway` column (added
        # 2026-05) and fall back to `on_ground` for older CSVs since
        # the recorder used to write the cleared runway only after touchdown.
        ils = (row.get('ils_runway') or '').strip()
        if not ils:
            ils = (row.get('on_ground') or '').strip()
        if ils:
            ac.ils_runway = ils
        ac.loc_intercepted = _bool_from_csv(row.get('loc'))
        ac.gs_intercepted = _bool_from_csv(row.get('gs'))

        on_ground = (row.get('on_ground') or '').strip()
        if on_ground:
            ac.on_ground = on_ground

        sim.aircraft_list[cs] = ac
        n += 1

    return n


def freeze_spawning(sim: SimulationEnv) -> None:
    """Monkey-patch the sim's spawner so no further aircraft enter.

    Used for handoff rollouts where the comparison should depend only
    on the planes present at the freeze frame.
    """
    sim.spawner.update = lambda delta_t: None
    sim.spawn_rate = 10**9
