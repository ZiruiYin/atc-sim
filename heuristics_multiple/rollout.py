"""Per-plane independent rollout module.

Snapshots the live sim, then for each armed plane runs K stochastic
rollouts in a fresh `SimulationEnv` with only that plane present and
spawning frozen. Keeps only LANDED trajectories.

This is the data layer for the heuristic-search method that will pick
non-intersecting paths across planes in a follow-up. By rolling each
plane out alone:
  - Each plane has K "what would I do if I were the only one here" options.
  - The user can later try mix-and-match — pick a path per plane such
    that they don't conflict in time/space — without the search space
    blowing up across 10^N joint rollouts.

Stochasticity across the K rollouts comes from temporarily renaming
the plane to `{cs}_R{k}` inside each sub-sim — `Runtime.tick` seeds its
mixture-noise generator from `hash(callsign)`, so a different name =
a different sample path.

Two execution modes:
  - Serial (default): `rollout_per_plane(...)` runs in-process. Cheap
    for low K + few planes, slow for big batches.
  - Parallel: call `init_pool(runtime, n_workers)` once at startup, then
    use `rollout_per_plane_parallel(...)`. Each worker process loads its
    own copy of the policy and runs rollouts independently — linear
    speedup up to `n_workers`. Pool persists across calls.
"""
from __future__ import annotations

from environment import SimulationEnv
from rl_multiple.handoff import reconstruct_from_frame, freeze_spawning


# ============================================================================ #
# Snapshot helpers
# ============================================================================ #


def _snapshot_row(sim, ac, cs):
    """Build a reconstruct_from_frame-compatible row from a live aircraft.

    Mirrors the column layout in `environment.core.human_data_logger.FIELDNAMES`
    so `reconstruct_from_frame` can consume it without any adaptation.
    """
    nmpp = sim.nm_per_pixel
    ax = sim.airport_x
    ay = sim.airport_y
    x_nm = (ac.x - ax) * nmpp
    y_nm = -(ac.y - ay) * nmpp
    return {
        'callsign': cs,
        'x_nm': str(x_nm),
        'y_nm': str(y_nm),
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
        # Armed planes have already had their STAR cleared on takeover —
        # no need to preserve.
        'star': '',
        'target_wpt': '',
        'terminal': '',
    }


def _is_landed(sub_sim, ac_obj, num_landed_before):
    """A plane counts as landed if either its `landed` flag is set OR the
    sim's `num_landed` counter ticked up (covers the case where the plane
    is removed from aircraft_list on touchdown in the same tick).
    """
    if ac_obj is not None and getattr(ac_obj, 'landed', False):
        return True
    if getattr(sub_sim, 'num_landed', 0) > num_landed_before:
        return True
    return False


def _build_subsim_with_plane(snap_row, rollout_cs, t0,
                              airport, radar_side):
    """Build a fresh sub-sim with one reconstructed plane, no spawning."""
    sub_sim = SimulationEnv(
        radar_side=radar_side, airport_name=airport,
        star_mode=True, spawn_single=False,
    )
    sub_sim.set_spawn_rate(10 ** 9)
    freeze_spawning(sub_sim)
    row = dict(snap_row)
    row['callsign'] = rollout_cs
    n = reconstruct_from_frame(sub_sim, {'sim_time': t0, 'rows': [row]})
    if n == 0:
        return None, None
    return sub_sim, sub_sim.aircraft_list[rollout_cs]


FINAL_APPROACH_KT = 140.0
# 3-degree glideslope rule of thumb: 300 ft/nm. Same formula as the
# training env's `_gs_capturable` (rl_multiple/multi_env.py:278).
GS_FT_PER_NM = 300.0
# Slack above the GS plane at LOC capture before we call it a failed
# intercept. Matches the training env default (`gs_capture_buffer_ft`).
GS_CAPTURE_BUFFER_FT = 50.0
# Rollout-only ground-deceleration bonus (kt/s) added on top of the
# base GROUND_DECELERATION_RATE=3. Total effective decel in the
# rollout sim becomes 6 kt/s, modelling the "runway clear" point at
# taxi speed rather than full stop — keeps closely-spaced landings
# from piling up on the runway in plans. Live sim is untouched.
ROLLOUT_GROUND_DECEL_BONUS_KT = 3.0


def _loc_capture_outcome(ac, runtime, sub_sim) -> str:
    """Classify a LOC-capture moment. Mirrors the training env's logic
    so rollouts drop failures the policy would otherwise be penalized for.

    Returns:
        'OK'             — captured before threshold, at-or-below GS+buffer
        'LOC_BEHIND_THR' — captured past the runway threshold (going wrong way)
        'LOC_ABOVE_GS'   — captured but too high to ever catch the glideslope
    """
    import math
    nmpp = sub_sim.nm_per_pixel
    thr_coords = ac.coords.get(runtime.runway)
    if thr_coords is None:
        return 'OK'
    dx_px = ac.x - thr_coords['x']
    dy_px = ac.y - thr_coords['y']
    d_thr_nm = nmpp * math.sqrt(dx_px * dx_px + dy_px * dy_px)
    gs_alt_ft = d_thr_nm * GS_FT_PER_NM

    ax = sub_sim.airport_x
    ay = sub_sim.airport_y
    x_nm = (ac.x - ax) * nmpp
    y_nm = -(ac.y - ay) * nmpp
    phi = math.radians((runtime.geom.course_deg + 180.0) % 360.0)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    dx = x_nm - runtime.geom.thr_x_nm
    dy = y_nm - runtime.geom.thr_y_nm
    a_along = dx * sin_phi + dy * cos_phi
    if a_along < 0.0:
        return 'LOC_BEHIND_THR'
    if float(ac.altitude) > gs_alt_ft + GS_CAPTURE_BUFFER_FT:
        return 'LOC_ABOVE_GS'
    return 'OK'


def _run_rollout(sub_sim, ac_obj, rollout_cs, runtime, max_steps):
    """Tick `runtime` against `sub_sim` until termination or `max_steps`.

    Returns `(points, outcome)` where outcome is one of:
        - 'LANDED'         landed cleanly within max_steps        — KEEP
        - 'INTERMEDIATE'   max_steps hit, plane still on a normal
                           approach (no explicit failure seen)    — KEEP
        - 'LOC_ABOVE_GS'   captured LOC too high to catch GS      — DROP
        - 'LOC_BEHIND_THR' captured LOC past the runway threshold — DROP
        - 'IMPROPER_EXIT'  plane removed without landing          — DROP

    `points` is the per-tick trajectory for keepable outcomes, `None` for
    failures. Caller is responsible for the keep/drop policy and for
    re-rolling on failure.

    Rollout-only environment overlay (NOT applied in the live watch sim):
    once the plane intercepts LOC, force `target_airspeed = 140 kt`. The
    base sim won't touch the policy's chosen approach speed; this
    overlay commits the rollout to final-approach speed.
    """
    try:
        runtime.state_for(rollout_cs).cleared = True
    except Exception:
        pass

    def _record(t, ac):
        # Full state snapshot — enough to deterministically replay this
        # trajectory in any sim by directly setting the aircraft fields,
        # AND to run conflict detection without rerunning physics.
        return {
            't': t,
            'x': float(ac.x), 'y': float(ac.y),
            'alt': float(ac.altitude),
            'hdg': float(ac.heading),
            'spd': float(ac.airspeed),
            'target_hdg': float(ac.target_heading),
            'target_alt': float(ac.target_altitude),
            'target_spd': float(ac.target_airspeed),
            'loc': bool(ac.loc_intercepted),
            'gs':  bool(ac.gs_intercepted),
            'on_ground': str(ac.on_ground) if ac.on_ground else '',
            'ils_runway': ac.ils_runway or '',
            'landed': bool(getattr(ac, 'landed', False)),
        }

    armed_local = {rollout_cs}
    points = [_record(0, ac_obj)]

    n_landed_before = getattr(sub_sim, 'num_landed', 0)
    prev_loc = bool(ac_obj.loc_intercepted)
    for step_i in range(1, max_steps + 1):
        runtime.tick(sub_sim, armed=armed_local)
        ac_pre = sub_sim.aircraft_list.get(rollout_cs)
        if ac_pre is not None:
            if ac_pre.loc_intercepted:
                # Hard final-approach speed override (after policy tick,
                # before physics step so the decel takes effect this tick).
                ac_pre.target_airspeed = FINAL_APPROACH_KT
            if ac_pre.on_ground and ac_pre.airspeed > 0:
                # Rollout-only ground-decel bonus: shaves the runway
                # occupation time roughly in half so plans don't pile
                # up at the threshold. Live sim is untouched.
                ac_pre.airspeed = max(
                    0.0,
                    ac_pre.airspeed - ROLLOUT_GROUND_DECEL_BONUS_KT)
        sub_sim.step(1.0)
        ac_now = sub_sim.aircraft_list.get(rollout_cs)

        # LOC-capture transition: classify and bail on failure.
        if ac_now is not None:
            loc_now = bool(ac_now.loc_intercepted)
            if loc_now and not prev_loc:
                cap_outcome = _loc_capture_outcome(ac_now, runtime, sub_sim)
                if cap_outcome != 'OK':
                    return None, cap_outcome
            prev_loc = loc_now

        if ac_now is not None:
            points.append(_record(step_i, ac_now))
        if _is_landed(sub_sim, ac_now, n_landed_before):
            return points, 'LANDED'
        if ac_now is None:
            # Removed without landing -> IMPROPER_EXIT.
            return None, 'IMPROPER_EXIT'

    # Loop exhausted: plane is still flying happily, we just ran out of
    # budget. KEEP the partial trajectory — the heuristic search cares
    # about where the plane was during the window, not just final state.
    return points, 'INTERMEDIATE'


# ============================================================================ #
# Serial executor — runs in the Flask process, blocks /rollout.
# ============================================================================ #


def rollout_per_plane(sim_live, runtime, armed,
                      n_per_plane: int = 1, max_steps: int = 5000,
                      max_attempts_per_slot: int = 5,
                      airport: str = 'test', runway: str = '27',
                      radar_side: int = 800) -> dict:
    """Sequential rollout — for small batches or debugging.

    Per plane, attempt up to `n_per_plane * max_attempts_per_slot`
    rollouts, keeping the first `n_per_plane` that don't end in an
    explicit failure (LOC_ABOVE_GS / LOC_BEHIND_THR / IMPROPER_EXIT).
    Rollouts that hit `max_steps` without landing are KEPT (treated as
    intermediate, not failure).
    """
    armed_set = {cs for cs in armed if cs in sim_live.aircraft_list}
    t0 = float(sim_live.sim_time)

    out: dict = {}
    for cs in armed_set:
        ac_live = sim_live.aircraft_list[cs]
        row = _snapshot_row(sim_live, ac_live, cs)
        origin_xy = {'x': float(ac_live.x), 'y': float(ac_live.y),
                     'sim_time': t0}

        kept: list = []
        attempt = 0
        cap = n_per_plane * max_attempts_per_slot
        n_landed = 0
        outcomes_seen: dict = {}
        while len(kept) < n_per_plane and attempt < cap:
            rollout_cs = f"{cs}_R{attempt}"
            attempt += 1
            sub_sim, ac_obj = _build_subsim_with_plane(
                row, rollout_cs, t0, airport, radar_side)
            if sub_sim is None:
                continue
            traj, outcome = _run_rollout(sub_sim, ac_obj, rollout_cs,
                                         runtime, max_steps)
            try:
                runtime.forget(rollout_cs)
            except Exception:
                pass
            outcomes_seen[outcome] = outcomes_seen.get(outcome, 0) + 1
            if traj is not None:
                kept.append(traj)
                if outcome == 'LANDED':
                    n_landed += 1

        out[cs] = {
            'origin': origin_xy,
            'rollouts': kept,
            'n_attempted': attempt,
            'n_kept': len(kept),
            'n_landed': n_landed,
            'outcomes': outcomes_seen,
        }

    return out


# ============================================================================ #
# Parallel executor — persistent process pool, init once at watch startup.
#
# Why processes (not threads): each `runtime.tick` is Python-heavy
# (encoding, sampling, command translation, sim.step). Threads can't
# overlap Python work. With `spawn` (required on Windows), each worker
# is a clean Python interpreter that loads the model once at init and
# then handles rollouts. Linear scaling up to n_workers.
# ============================================================================ #


# Worker-process globals (only ever populated inside worker processes).
_worker_runtime = None

# Parent-process globals — the live `_pool` handle + its config signature
# so we don't reinit on every /rollout call.
_pool = None
_pool_signature: tuple | None = None


def _worker_init(ckpt_path, bc_seed_path, runway, alt_floor_ft,
                 issue_speed, deterministic):
    """Run once per worker process at pool startup. Loads the policy."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from rl_multiple.runtime import Runtime
    global _worker_runtime
    _worker_runtime = Runtime(
        ckpt_path=ckpt_path,
        bc_seed_path=bc_seed_path,
        runway=runway,
        alt_floor_ft=alt_floor_ft,
        issue_speed=issue_speed,
        deterministic=deterministic,
    )


def _worker_rollout_task(args):
    """Run one rollout. Returns `(points_or_None, outcome)` so the caller
    can decide keep/drop and retry.

    Args tuple kept flat for easy pickling.
    """
    snap_row, rollout_cs, t0, max_steps, airport, radar_side = args
    try:
        sub_sim, ac_obj = _build_subsim_with_plane(
            snap_row, rollout_cs, t0, airport, radar_side)
        if sub_sim is None:
            return (None, 'NO_SUB_SIM')
        return _run_rollout(sub_sim, ac_obj, rollout_cs,
                            _worker_runtime, max_steps)
    except Exception:
        import traceback
        traceback.print_exc()
        return None  # signal worker error so caller can categorize


def _runtime_signature(runtime) -> tuple:
    return (
        runtime._ppo_ckpt_path,
        runtime._bc_seed_path,
        runtime.runway,
        runtime.alt_floor_ft,
        runtime.issue_speed,
        runtime.deterministic,
    )


def _warmup_noop(_unused):
    """Marker task that returns once a worker has finished its initializer."""
    return True


def init_pool(runtime, n_workers: int = 6, warm: bool = True):
    """Spin up (or reuse) a persistent worker pool. Returns the pool.

    Idempotent for the same runtime config — subsequent calls with the
    same checkpoint return the existing pool without re-spawning.

    If `warm=True`, blocks until each worker has completed its
    initializer (loaded the policy). This pushes the spawn-import +
    model-load cost out of the first /rollout call. On Windows + spawn
    that's typically a 3–6 s one-time penalty per worker (parallelized).
    """
    import multiprocessing as mp
    global _pool, _pool_signature
    sig = _runtime_signature(runtime)
    if _pool is not None and _pool_signature == sig:
        return _pool
    shutdown_pool()
    ctx = mp.get_context('spawn')
    _pool = ctx.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=sig,
    )
    _pool_signature = sig
    if warm:
        # Dispatch n_workers no-op tasks. The pool round-robins, so each
        # worker handles one and we know all initializers have run by
        # the time map returns.
        _pool.map(_warmup_noop, list(range(n_workers)))
    return _pool


def shutdown_pool():
    """Tear down the persistent pool, if any. Safe to call repeatedly."""
    global _pool, _pool_signature
    if _pool is not None:
        try:
            _pool.terminate()
            _pool.join()
        except Exception:
            pass
        _pool = None
        _pool_signature = None


def rollout_per_plane_parallel(sim_live, runtime, armed,
                                n_per_plane: int = 1, max_steps: int = 5000,
                                max_attempts_per_slot: int = 5,
                                airport: str = 'test', runway: str = '27',
                                radar_side: int = 800) -> dict:
    """Parallel rollout with retry-on-failure.

    Each round dispatches one task per outstanding slot (plane × needed
    trajectory) and collects results. Failures (LOC_ABOVE_GS,
    LOC_BEHIND_THR, IMPROPER_EXIT) trigger another attempt in the next
    round; LANDED and INTERMEDIATE (max_steps without explicit failure)
    are both KEPT. Loops up to `max_attempts_per_slot` rounds — so the
    worst case is `n_per_plane * max_attempts_per_slot * n_planes` tasks
    dispatched, but most planes finish in 1 round.

    Falls back to serial if `init_pool` hasn't been called.
    """
    if _pool is None:
        return rollout_per_plane(
            sim_live, runtime, armed,
            n_per_plane=n_per_plane, max_steps=max_steps,
            max_attempts_per_slot=max_attempts_per_slot,
            airport=airport, runway=runway, radar_side=radar_side)

    armed_set = {cs for cs in armed if cs in sim_live.aircraft_list}
    t0 = float(sim_live.sim_time)

    out: dict = {}
    rows: dict = {}
    next_suffix: dict = {}
    for cs in armed_set:
        ac_live = sim_live.aircraft_list[cs]
        rows[cs] = _snapshot_row(sim_live, ac_live, cs)
        next_suffix[cs] = 0
        out[cs] = {
            'origin': {'x': float(ac_live.x), 'y': float(ac_live.y),
                       'sim_time': t0},
            'rollouts': [],
            'n_attempted': 0,
            'n_kept': 0,
            'n_landed': 0,
            'outcomes': {},
        }

    if not armed_set:
        return out

    for round_i in range(max_attempts_per_slot):
        tasks: list = []
        task_owner: list = []
        for cs in armed_set:
            need = n_per_plane - out[cs]['n_kept']
            for _ in range(need):
                k = next_suffix[cs]
                next_suffix[cs] += 1
                rollout_cs = f"{cs}_R{k}"
                tasks.append((dict(rows[cs]), rollout_cs, t0, max_steps,
                              airport, radar_side))
                task_owner.append(cs)
        if not tasks:
            break

        results = _pool.map(_worker_rollout_task, tasks)

        for cs_task, result in zip(task_owner, results):
            out[cs_task]['n_attempted'] += 1
            if result is None:
                # Worker crash — treat as failure.
                outcome = 'WORKER_ERROR'
                traj = None
            else:
                traj, outcome = result
            out[cs_task]['outcomes'][outcome] = (
                out[cs_task]['outcomes'].get(outcome, 0) + 1)
            if traj is not None:
                out[cs_task]['rollouts'].append(traj)
                out[cs_task]['n_kept'] += 1
                if outcome == 'LANDED':
                    out[cs_task]['n_landed'] += 1

    return out
