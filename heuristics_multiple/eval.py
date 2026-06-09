"""Plan-mode eval driver — collects N kept trajectories produced by the
flight-plan heuristic.

Per scenario:
  1. Spin up a fresh `SimulationEnv` with the strict 2 NM / 1000 ft
     same-medium collision rule installed.
  2. Run the watch's plan-mode loop synchronously (no Flask, no async
     thread): arm planes after warmup, replan when needed via
     `flight_plan.replan_all`, force plan state each tick, runtime.tick
     for any plane without a plan, sim.step.
  3. Apply the per-plane lifetime cap (`plan_steps`) — any plane armed
     for > plan_steps without finishing is force-removed and recorded
     as TIMEOUT.
  4. Stop on crash or when max_steps is reached.

Outcome categories (per plane):
  LANDED        — touched down on runway
  CRASHED       — terminating collision pair
  IMPROPER_EXIT — flew out of radar bounds
  TIMEOUT       — wiped by the lifetime cap (= armed for > plan_steps
                  without landing or exiting)
  TRUNCATED    — sim stopped before this plane finished naturally;
                 dropped before metrics

Metrics (per the user's REPORT.md spec):
  policy_sr      = LANDED / kept       where kept = LANDED + CRASHED
                                             + IMPROPER_EXIT + TIMEOUT
  num_crashes    = absolute count of CRASHED in kept
  avg_violation_s = sum(violation_s over kept) / kept
"""
from __future__ import annotations

import random
import sys
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.core.simulation import SimulationEnv  # noqa: E402
from environment.core.human_data_logger import HumanDataRecorder  # noqa: E402
from environment.utils import distance_between_coords_pixels  # noqa: E402
from environment.params import TRAJ_LENGTH  # noqa: E402

from heuristics_multiple.flight_plan import replan_all, FlightPlan  # noqa: E402


# --------------------------------------------------------------------------- #
# Strict pair-check rule — same as compare_eval / watch.
# --------------------------------------------------------------------------- #
WARN_LATERAL_NM = 2.0
WARN_VERTICAL_FT = 1000.0


def _check_aircraft_pair_strict(self, aircraft1, aircraft2):
    pixel_distance = distance_between_coords_pixels(
        aircraft1.x, aircraft1.y, aircraft2.x, aircraft2.y)
    lateral_nm = pixel_distance * self.nm_per_pixel
    vertical_separation = abs(aircraft1.altitude - aircraft2.altitude)
    same_medium = bool(aircraft1.on_ground) == bool(aircraft2.on_ground)
    if (same_medium and lateral_nm < WARN_LATERAL_NM
            and vertical_separation < WARN_VERTICAL_FT):
        aircraft1.collision_warning = True
        aircraft2.collision_warning = True
    if vertical_separation <= 50:
        crash_threshold_pixels = 0.2 / self.nm_per_pixel
        if pixel_distance <= crash_threshold_pixels:
            aircraft1.crash = f"collided with {aircraft2.callsign}"
            aircraft2.crash = f"collided with {aircraft1.callsign}"


def _install_strict_collision_rule(sim_obj):
    sim_obj.collision_monitor._check_aircraft_pair = types.MethodType(
        _check_aircraft_pair_strict, sim_obj.collision_monitor)


@dataclass
class Traj:
    callsign: str
    outcome: str
    spawn_t: float
    term_t: float
    violation_s: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ScenarioResult:
    seed: int
    crash_time: Optional[float]
    sim_time_end: float
    trajs: list = field(default_factory=list)
    csv_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# Plan-mode helpers (synchronous, no Flask). Mirror watch.py's per-tick logic.
# --------------------------------------------------------------------------- #


def _arm_plane(sim, runtime, cs, ac, runway: str,
                armed_at: dict, plane_star: dict) -> None:
    # Capture the STAR name BEFORE clearing it — needed for the
    # pre-LOC timeout cap, which depends on STAR class (1/2 vs 3).
    if ac.star_name:
        plane_star[cs] = ac.star_name
    if ac.star is not None or ac.target_wpt is not None:
        ac.star = None
        ac.star_name = None
        ac.target_wpt = None
    res = sim.command(cs, f"L {runway}")
    state_for = getattr(runtime, 'state_for', None)
    if state_for and res.get('ok'):
        state_for(cs).cleared = True
    if cs not in armed_at:
        armed_at[cs] = float(sim.sim_time)


def _check_arm(sim, runtime, runway, armed, armed_at, initial_star_len,
               plane_star, warmup_wpts=2):
    for cs, ac in sim.aircraft_list.items():
        if cs in armed:
            continue
        if cs not in initial_star_len:
            initial_star_len[cs] = len(ac.star) if ac.star else 0
            continue
        initial = initial_star_len[cs]
        current = len(ac.star) if ac.star else 0
        popped = initial - current
        threshold = min(warmup_wpts, initial) if initial > 0 else 0
        if initial == 0 or popped >= threshold:
            _arm_plane(sim, runtime, cs, ac, runway, armed_at, plane_star)
            armed.add(cs)


def _active_set(sim, armed):
    return {cs for cs in armed
            if cs in sim.aircraft_list
            and not sim.aircraft_list[cs].on_ground}


def _replan_needed(active, flight_plans) -> bool:
    if not active:
        return False
    if not flight_plans:
        return True
    for cs in active:
        if cs not in flight_plans:
            return True
    if all(flight_plans[cs].depleted for cs in active
           if cs in flight_plans):
        return True
    return False


def _apply_flight_plans(sim, flight_plans):
    if not flight_plans:
        return
    for cs in list(flight_plans.keys()):
        plan = flight_plans[cs]
        ac = sim.aircraft_list.get(cs)
        if ac is None:
            del flight_plans[cs]
            continue
        if plan.depleted:
            if hasattr(ac, '_plan_orig_update'):
                ac.update = ac._plan_orig_update
                delattr(ac, '_plan_orig_update')
            del flight_plans[cs]
            continue
        s = plan.head_state()
        ac.x = float(s['x'])
        ac.y = float(s['y'])
        ac.altitude = float(s['alt'])
        ac.heading = float(s['hdg'])
        ac.airspeed = float(s['spd'])
        ac.target_heading = float(s['target_hdg'])
        ac.target_altitude = float(s['target_alt'])
        ac.target_airspeed = float(s['target_spd'])
        og = s.get('on_ground') or None
        ac.on_ground = og if og else None
        if og:
            ac.altitude = 0.0
            ac.ils_runway = None
            ac.loc_intercepted = False
            ac.gs_intercepted = False
            ac.target_altitude = 0.0
            ac.target_airspeed = 0.0
        else:
            ac.loc_intercepted = bool(s['loc'])
            ac.gs_intercepted = bool(s['gs'])
            ils = s.get('ils_runway') or None
            if ils:
                ac.ils_runway = ils
        if s.get('landed'):
            ac.landed = True
        ac.trajectory.append((ac.x, ac.y))
        if len(ac.trajectory) > TRAJ_LENGTH:
            ac.trajectory = ac.trajectory[-TRAJ_LENGTH:]
        if not hasattr(ac, '_plan_orig_update'):
            ac._plan_orig_update = ac.update
            ac.update = types.MethodType(lambda self, dt: None, ac)
        plan.advance()


def _enforce_pre_loc_timeout(sim, armed, plane_star, pre_loc_steps,
                              max_steps_1_2, max_steps_3,
                              flight_plans, armed_at, initial_star_len,
                              plane_spawn_t, plane_violation_s,
                              completions, completed_cs, runtime):
    """Pre-LOC step cap — mirrors the training env's TIMEOUT rule
    (`rl_multiple/multi_env.py:645-655`).

    For each ARMED plane that hasn't intercepted LOC yet, increment its
    pre-LOC step counter. If the counter exceeds the STAR-specific cap
    (max_steps_3 for STAR ending in '3', else max_steps_1_2), force-wipe
    the plane and record it as TIMEOUT. Post-LOC ticks don't count and
    don't get the plane wiped — once the plane has captured LOC, it has
    as long as it needs to finish the approach.
    """
    now = float(sim.sim_time)
    to_wipe = []
    for cs in list(armed):
        if cs not in sim.aircraft_list:
            continue
        ac = sim.aircraft_list[cs]
        # Post-LOC: stop accumulating, no wipe possible.
        if ac.loc_intercepted:
            continue
        pre_loc_steps[cs] = pre_loc_steps.get(cs, 0) + 1
        star = plane_star.get(cs, '')
        cap = max_steps_3 if star.endswith('3') else max_steps_1_2
        if pre_loc_steps[cs] >= cap:
            to_wipe.append(cs)
    if not to_wipe:
        return
    for cs in to_wipe:
        ac = sim.aircraft_list.get(cs)
        if ac is not None:
            if hasattr(ac, '_plan_orig_update'):
                ac.update = ac._plan_orig_update
                delattr(ac, '_plan_orig_update')
            try:
                # NOTE: do NOT bump sim.improper_exits.
                del sim.aircraft_list[cs]
            except Exception:
                pass
            completions.append(Traj(
                callsign=cs, outcome='TIMEOUT',
                spawn_t=plane_spawn_t.get(cs, now),
                term_t=now,
                violation_s=plane_violation_s.get(cs, 0.0)))
            completed_cs.add(cs)
        armed_at.pop(cs, None)
        flight_plans.pop(cs, None)
        armed.discard(cs)
        initial_star_len.pop(cs, None)
        pre_loc_steps.pop(cs, None)
        plane_star.pop(cs, None)
        if runtime is not None:
            try:
                runtime.forget(cs)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Single scenario runner.
# --------------------------------------------------------------------------- #


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_scenario(runtime,
                     seed: int,
                     spawn_rate: int = 90,
                     max_steps: int = 1500,
                     plan_steps: int = 500,
                     max_steps_1_2: int = 1200,
                     max_steps_3: int = 500,
                     batch_size: int = 3,
                     max_conflict_iters: int = 3,
                     airport: str = 'test',
                     runway: str = '27',
                     warmup_wpts: int = 2,
                     out_csv: Optional[Path] = None,
                     stop_after_kept: Optional[int] = None,
                     full_rollouts: bool = False,
                     full_rollout_max_steps: int = 5000) -> ScenarioResult:
    """Run one plan-mode scenario. Returns trajectories + crash time."""
    _seed_all(seed)
    if hasattr(runtime, 'reset'):
        runtime.reset()
    sim = SimulationEnv(radar_side=800, airport_name=airport,
                        spawn_rate=spawn_rate, star_mode=True)
    _install_strict_collision_rule(sim)

    if out_csv is not None:
        rec = HumanDataRecorder(spawn_single=False, in_memory=True)
        rec.start()
        sim.recorder = rec

    # Per-callsign bookkeeping.
    plane_spawn_t: dict = {}
    plane_violation_s: dict = {}
    completions: list = []
    completed_cs: set = set()
    crashed_callsigns: set = set()
    armed: set = set()
    armed_at: dict = {}
    plane_star: dict = {}      # cs -> STAR name (e.g. 'NORTH3')
    pre_loc_steps: dict = {}    # cs -> ticks since arming, pre-LOC only
    initial_star_len: dict = {}
    flight_plans: dict = {}

    crash_time: Optional[float] = None

    for _ in range(max_steps):
        # Pre-step bookkeeping.
        for cs, ac in sim.aircraft_list.items():
            if cs not in plane_spawn_t:
                plane_spawn_t[cs] = sim.sim_time
                plane_violation_s[cs] = 0.0
            if ac.collision_warning:
                plane_violation_s[cs] += 1.0

        pre_keys = set(sim.aircraft_list.keys())
        pre_landed, pre_exit = sim.num_landed, sim.improper_exits

        _check_arm(sim, runtime, runway, armed, armed_at,
                   initial_star_len, plane_star, warmup_wpts)
        _enforce_pre_loc_timeout(
            sim, armed, plane_star, pre_loc_steps,
            max_steps_1_2, max_steps_3,
            flight_plans, armed_at, initial_star_len,
            plane_spawn_t, plane_violation_s,
            completions, completed_cs, runtime)

        # Replan synchronously when needed.
        active = _active_set(sim, armed)
        if _replan_needed(active, flight_plans):
            # Restore monkey-patches from any old plans first so the
            # replan thread doesn't operate on no-op'd aircraft.
            for cs, plan in flight_plans.items():
                ac = sim.aircraft_list.get(cs)
                if ac is not None and hasattr(ac, '_plan_orig_update'):
                    ac.update = ac._plan_orig_update
                    delattr(ac, '_plan_orig_update')
            new_plans, _residual = replan_all(
                sim_live=sim, runtime=runtime,
                active_planes=active,
                existing_plans=flight_plans,
                plan_steps=plan_steps,
                max_conflict_iters=max_conflict_iters,
                batch_size=batch_size,
                airport=airport, radar_side=sim.radar_side,
                full_rollouts=full_rollouts,
                full_rollout_max_steps=full_rollout_max_steps)
            flight_plans = new_plans

        _apply_flight_plans(sim, flight_plans)

        armed_no_plan = armed - set(flight_plans.keys())
        runtime.tick(sim, armed=armed_no_plan)
        sim.step(1.0)

        if sim.crash_occurred:
            for cs, ac in sim.aircraft_list.items():
                if ac.crash and cs not in completed_cs:
                    crashed_callsigns.add(cs)
                    completions.append(Traj(
                        callsign=cs, outcome='CRASHED',
                        spawn_t=plane_spawn_t.get(cs, sim.sim_time),
                        term_t=sim.sim_time,
                        violation_s=plane_violation_s.get(cs, 0.0)))
                    completed_cs.add(cs)
            crash_time = sim.sim_time
            break

        # Early stop when we have enough kept (LANDED/CRASHED/IMPROPER_EXIT/
        # TIMEOUT) trajectories. Doesn't drop the still-airborne planes —
        # they just become TRUNCATED in the wrap-up below.
        if stop_after_kept is not None:
            kept_count = sum(1 for t in completions
                             if t.outcome not in ('TRUNCATED', 'UNKNOWN'))
            if kept_count >= stop_after_kept:
                break

        # Attribute removals from THIS sim.step (TIMEOUT removals from
        # _enforce_lifetime_limit are excluded via completed_cs).
        post_keys = set(sim.aircraft_list.keys())
        removed = sorted((pre_keys - post_keys) - completed_cs)
        d_landed = sim.num_landed - pre_landed
        d_exit = sim.improper_exits - pre_exit
        outcomes = (['LANDED'] * d_landed
                    + ['IMPROPER_EXIT'] * d_exit
                    + ['UNKNOWN'] * max(0, len(removed) - d_landed - d_exit))
        for cs, outcome in zip(removed, outcomes):
            completions.append(Traj(
                callsign=cs, outcome=outcome,
                spawn_t=plane_spawn_t.get(cs, sim.sim_time),
                term_t=sim.sim_time,
                violation_s=plane_violation_s.get(cs, 0.0)))
            completed_cs.add(cs)
            flight_plans.pop(cs, None)
            armed.discard(cs)
            armed_at.pop(cs, None)
            initial_star_len.pop(cs, None)
            plane_star.pop(cs, None)
            pre_loc_steps.pop(cs, None)

    # Truncated: anything still alive at scenario end.
    for cs in sim.aircraft_list:
        if cs in crashed_callsigns:
            continue
        completions.append(Traj(
            callsign=cs, outcome='TRUNCATED',
            spawn_t=plane_spawn_t.get(cs, sim.sim_time),
            term_t=sim.sim_time,
            violation_s=plane_violation_s.get(cs, 0.0)))

    if sim.recorder is not None:
        if out_csv is not None:
            csv_text = sim.recorder.to_csv()
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text(csv_text, encoding='utf-8')
        sim.recorder.close()
        sim.recorder = None

    return ScenarioResult(
        seed=seed,
        crash_time=crash_time,
        sim_time_end=sim.sim_time,
        trajs=completions,
        csv_path=str(out_csv) if out_csv else None,
    )


# --------------------------------------------------------------------------- #
# Metrics — match the REPORT.md spec.
# --------------------------------------------------------------------------- #


def keep_outcome(o: str) -> bool:
    """Per the report: TRUNCATED is dropped before metrics. Everything
    else (LANDED / CRASHED / IMPROPER_EXIT / TIMEOUT / LOC_ABOVE_GS)
    is kept."""
    return o != 'TRUNCATED' and o != 'UNKNOWN'


def compute_metrics(kept_trajs: list) -> dict:
    n = len(kept_trajs)
    if n == 0:
        return {'n': 0, 'policy_sr': 0.0, 'num_crashes': 0,
                'avg_violation_s': 0.0,
                'n_landed': 0, 'n_crashed': 0,
                'n_improper_exit': 0, 'n_timeout': 0,
                'n_loc_above_gs': 0}
    counts = {}
    sum_violation = 0.0
    for t in kept_trajs:
        counts[t.outcome] = counts.get(t.outcome, 0) + 1
        sum_violation += t.violation_s
    n_landed = counts.get('LANDED', 0)
    n_crashed = counts.get('CRASHED', 0)
    n_exit = counts.get('IMPROPER_EXIT', 0)
    n_timeout = counts.get('TIMEOUT', 0)
    n_loc_above_gs = counts.get('LOC_ABOVE_GS', 0)
    return {
        'n': n,
        'policy_sr': n_landed / n,
        'num_crashes': n_crashed,
        'avg_violation_s': sum_violation / n,
        'n_landed': n_landed,
        'n_crashed': n_crashed,
        'n_improper_exit': n_exit,
        'n_timeout': n_timeout,
        'n_loc_above_gs': n_loc_above_gs,
    }
