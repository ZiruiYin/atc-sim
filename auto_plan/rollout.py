"""Per-plane rollout layer for the AUTO planner.

Snapshots a live aircraft, rebuilds it alone in a fresh frozen-spawning
`SimulationEnv`, and ticks the GMM policy forward to produce a deterministic
trajectory (full per-tick state). Termination follows the SIMULATOR only:
LANDED (touched down), IMPROPER_EXIT (removed without landing), or
INTERMEDIATE (hit the planning horizon). Unlike the rl eval harness, this
does NOT classify or drop LOC-above-glideslope / behind-threshold outcomes —
the live sim decides every plane's fate.

Includes a self-contained port of the rl `handoff` helpers (reconstruct +
freeze spawning) so `auto_plan` has no dependency on the training packages,
plus an optional persistent worker pool (spawn) sized by the caller.
"""

from __future__ import annotations

from environment import SimulationEnv
from environment.core.aircraft import Aircraft
from environment.params import TRAJ_LENGTH


# ============================================================================ #
# Handoff helpers (ported from rl_multiple/handoff.py, trimmed to a single
# in-memory frame — no CSV).
# ============================================================================ #


def _float_or(v, default=0.0):
    try:
        return float(v) if v not in (None, '') else default
    except (TypeError, ValueError):
        return default


def _bool_from(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ('true', '1', 't', 'yes')


def reconstruct_from_frame(sim: SimulationEnv, frame: dict) -> int:
    """Wipe `sim` of current aircraft and rebuild from `frame['rows']`."""
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
            ac.target_heading = _float_or(row.get('target_heading'), default=ac.target_heading)
            ac.target_altitude = _float_or(row.get('target_altitude'), default=ac.target_altitude)
            ac.target_airspeed = _float_or(row.get('target_airspeed'), default=ac.target_airspeed)
        elif target_wpt:
            ac.target_wpt = target_wpt

        ils = (row.get('ils_runway') or '').strip()
        if not ils:
            ils = (row.get('on_ground') or '').strip()
        if ils:
            ac.ils_runway = ils
        ac.loc_intercepted = _bool_from(row.get('loc'))
        ac.gs_intercepted = _bool_from(row.get('gs'))

        on_ground = (row.get('on_ground') or '').strip()
        if on_ground:
            ac.on_ground = on_ground

        sim.aircraft_list[cs] = ac
        n += 1
    return n


def freeze_spawning(sim: SimulationEnv) -> None:
    """No new arrivals during a rollout."""
    sim.spawner.update = lambda delta_t: None
    sim.spawn_rate = 10 ** 9


# ============================================================================ #
# Snapshot + sub-sim construction
# ============================================================================ #


def _snapshot_row(sim, ac, cs):
    nmpp = sim.nm_per_pixel
    ax = sim.airport_x
    ay = sim.airport_y
    return {
        'callsign': cs,
        'x_nm': str((ac.x - ax) * nmpp),
        'y_nm': str(-(ac.y - ay) * nmpp),
        'altitude': str(ac.altitude),
        'heading': str(ac.heading),
        'airspeed': str(ac.airspeed),
        'target_altitude': str(ac.target_altitude),
        'target_heading': str(ac.target_heading),
        'target_airspeed': str(ac.target_airspeed),
        'loc': 'true' if ac.loc_intercepted else 'false',
        'gs': 'true' if ac.gs_intercepted else 'false',
        'on_ground': str(ac.on_ground) if ac.on_ground else '',
        'ils_runway': ac.ils_runway or '',
        'star': '', 'target_wpt': '', 'terminal': '',
    }


def _build_subsim_with_plane(snap_row, rollout_cs, t0, airport, radar_side):
    sub_sim = SimulationEnv(radar_side=radar_side, airport_name=airport,
                            star_mode=True, spawn_single=False)
    sub_sim.set_spawn_rate(10 ** 9)
    freeze_spawning(sub_sim)
    row = dict(snap_row)
    row['callsign'] = rollout_cs
    n = reconstruct_from_frame(sub_sim, {'sim_time': t0, 'rows': [row]})
    if n == 0:
        return None, None
    return sub_sim, sub_sim.aircraft_list[rollout_cs]


# ============================================================================ #
# Rollout loop — termination follows the simulator (no LOC-outcome dropping).
# ============================================================================ #

FINAL_APPROACH_KT = 140.0
ROLLOUT_GROUND_DECEL_BONUS_KT = 3.0


def _is_landed(sub_sim, ac_obj, num_landed_before):
    if ac_obj is not None and getattr(ac_obj, 'landed', False):
        return True
    if getattr(sub_sim, 'num_landed', 0) > num_landed_before:
        return True
    return False


def _record(t, ac):
    return {
        't': t,
        'x': float(ac.x), 'y': float(ac.y),
        'alt': float(ac.altitude), 'hdg': float(ac.heading),
        'spd': float(ac.airspeed),
        'target_hdg': float(ac.target_heading),
        'target_alt': float(ac.target_altitude),
        'target_spd': float(ac.target_airspeed),
        'loc': bool(ac.loc_intercepted), 'gs': bool(ac.gs_intercepted),
        'on_ground': str(ac.on_ground) if ac.on_ground else '',
        'ils_runway': ac.ils_runway or '',
        'landed': bool(getattr(ac, 'landed', False)),
    }


def _run_rollout(sub_sim, ac_obj, rollout_cs, runtime, max_steps):
    """Tick the policy until termination or `max_steps`. Returns
    `(points, outcome)`:
        'LANDED'        - touched down within horizon              (keep)
        'INTERMEDIATE'  - horizon hit, still flying                (keep)
        'IMPROPER_EXIT' - removed without landing -> points=None   (retry)

    Rollout-only overlays (NOT in the live sim): final-approach speed lock at
    LOC capture, and a ground-decel bonus so closely-spaced landings don't pile
    up on the runway in the plan.
    """
    try:
        runtime.state_for(rollout_cs).cleared = True
    except Exception:
        pass

    armed_local = {rollout_cs}
    points = [_record(0, ac_obj)]
    n_landed_before = getattr(sub_sim, 'num_landed', 0)

    for step_i in range(1, max_steps + 1):
        runtime.tick(sub_sim, armed=armed_local)
        ac_pre = sub_sim.aircraft_list.get(rollout_cs)
        if ac_pre is not None:
            if ac_pre.loc_intercepted:
                ac_pre.target_airspeed = FINAL_APPROACH_KT
            if ac_pre.on_ground and ac_pre.airspeed > 0:
                ac_pre.airspeed = max(0.0, ac_pre.airspeed - ROLLOUT_GROUND_DECEL_BONUS_KT)
        sub_sim.step(1.0)
        ac_now = sub_sim.aircraft_list.get(rollout_cs)

        if ac_now is not None:
            points.append(_record(step_i, ac_now))
        if _is_landed(sub_sim, ac_now, n_landed_before):
            return points, 'LANDED'
        if ac_now is None:
            return None, 'IMPROPER_EXIT'

    return points, 'INTERMEDIATE'


# ============================================================================ #
# Persistent worker pool (spawn) — sized by the caller (adaptive to CPU count).
# Each worker loads the policy once at init.
# ============================================================================ #

_worker_runtime = None
_pool = None
_pool_signature: tuple | None = None


def _worker_init(ckpt_path, config_path, runway, alt_floor_ft,
                 issue_speed, deterministic):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from auto_plan.runtime import Runtime
    global _worker_runtime
    _worker_runtime = Runtime(ckpt_path=ckpt_path, config_path=config_path,
                              runway=runway, alt_floor_ft=alt_floor_ft,
                              issue_speed=issue_speed, deterministic=deterministic)


def _worker_rollout_task(args):
    snap_row, rollout_cs, t0, max_steps, airport, radar_side = args
    try:
        sub_sim, ac_obj = _build_subsim_with_plane(
            snap_row, rollout_cs, t0, airport, radar_side)
        if sub_sim is None:
            return (None, 'NO_SUB_SIM')
        return _run_rollout(sub_sim, ac_obj, rollout_cs, _worker_runtime, max_steps)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _warmup_noop(_unused):
    return True


def init_pool(runtime, n_workers, warm=True):
    """Spin up (or reuse) a persistent spawn pool. `n_workers` is chosen by the
    caller (adaptive to CPU count). Returns the pool, or None for n_workers<=1
    (serial execution)."""
    if n_workers is None or n_workers <= 1:
        return None
    import multiprocessing as mp
    global _pool, _pool_signature
    sig = (runtime._ckpt_path, str(runtime.config.get('ckpt')), runtime.runway,
           runtime.alt_floor_ft, runtime.issue_speed, runtime.deterministic, n_workers)
    if _pool is not None and _pool_signature == sig:
        return _pool
    shutdown_pool()
    # config_path=None -> worker Runtime defaults to auto_plan/policy_config.json.
    initargs = (runtime._ckpt_path, None,
                runtime.runway, runtime.alt_floor_ft,
                runtime.issue_speed, runtime.deterministic)
    ctx = mp.get_context('spawn')
    _pool = ctx.Pool(n_workers, initializer=_worker_init, initargs=initargs)
    _pool_signature = sig
    if warm:
        _pool.map(_warmup_noop, list(range(n_workers)))
    return _pool


def get_pool():
    return _pool


def shutdown_pool():
    global _pool, _pool_signature
    if _pool is not None:
        try:
            _pool.terminate()
            _pool.join()
        except Exception:
            pass
        _pool = None
        _pool_signature = None
