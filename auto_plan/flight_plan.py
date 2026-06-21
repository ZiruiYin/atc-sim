"""Multi-plane flight planning with conflict resolution (AUTO planner).

For each active plane, roll out independent candidate trajectories, then pick
one candidate per plane such that no pair violates the separation cone
(2 NM lateral / 1000 ft vertical, same-medium gated) — by backtracking search,
falling back to greedy max-separation. Plans are full per-tick state lists; the
live sim replays them by directly forcing the aircraft each tick.

Ported and trimmed from the rl branch's `heuristics_multiple/flight_plan.py`.
The fixed-horizon planning path is kept; the LANDED-only "full rollout" eval
path is dropped. Worker-pool access goes through `rollout.get_pool()` so the
live pool handle is always current (the rl version imported `_pool` by value,
which left it stale at None).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from auto_plan import rollout as _ro
from auto_plan.rollout import (
    _snapshot_row, _worker_rollout_task, _build_subsim_with_plane, _run_rollout,
)

# Separation cone — identical to the live SIMULATED collision rule.
PLANNING_LATERAL_NM = 2.0
WARN_VERTICAL_FT = 1000.0


@dataclass
class FlightPlan:
    callsign: str
    t0_sim: float
    states: list
    cursor: int = 0
    outcome: str = ''
    attempt: int = 0

    @property
    def depleted(self) -> bool:
        return self.cursor >= len(self.states)

    @property
    def remaining(self) -> int:
        return max(0, len(self.states) - self.cursor)

    def head_state(self) -> dict | None:
        if self.depleted:
            return None
        return self.states[self.cursor]

    def advance(self) -> None:
        self.cursor += 1


# --------------------------------------------------------------------------- #
# Conflict detection over recorded per-tick state lists.
# --------------------------------------------------------------------------- #


def find_conflicts(plans: dict, nm_per_pixel: float, max_ticks: int) -> list:
    cs_list = list(plans.keys())
    pair_data: dict = {}
    for t in range(max_ticks + 1):
        for i in range(len(cs_list)):
            for j in range(i + 1, len(cs_list)):
                a, b = cs_list[i], cs_list[j]
                ta, tb = plans[a], plans[b]
                if t >= len(ta) or t >= len(tb):
                    continue
                sa, sb = ta[t], tb[t]
                if bool(sa.get('on_ground')) != bool(sb.get('on_ground')):
                    continue
                if abs(sa['alt'] - sb['alt']) >= WARN_VERTICAL_FT:
                    continue
                dx = sa['x'] - sb['x']
                dy = sa['y'] - sb['y']
                lateral_nm = math.sqrt(dx * dx + dy * dy) * nm_per_pixel
                if lateral_nm >= PLANNING_LATERAL_NM:
                    continue
                rec = pair_data.get((a, b))
                if rec is None:
                    pair_data[(a, b)] = {'a': a, 'b': b, 'first_t': t,
                                         'min_sep_nm': lateral_nm, 'min_sep_t': t}
                elif lateral_nm < rec['min_sep_nm']:
                    rec['min_sep_nm'] = lateral_nm
                    rec['min_sep_t'] = t
    return list(pair_data.values())


def _aligned_tail(plan: 'FlightPlan') -> list:
    return plan.states[plan.cursor:]


def _pair_conflict(plan_a, plan_b, nm_per_pixel, max_ticks):
    tail_a = _aligned_tail(plan_a)
    tail_b = _aligned_tail(plan_b)
    n = min(len(tail_a), len(tail_b), max_ticks + 1)
    min_sep = float('inf')
    min_sep_t = -1
    for t in range(n):
        sa, sb = tail_a[t], tail_b[t]
        if bool(sa.get('on_ground')) != bool(sb.get('on_ground')):
            continue
        if abs(sa['alt'] - sb['alt']) >= WARN_VERTICAL_FT:
            continue
        dx = sa['x'] - sb['x']
        dy = sa['y'] - sb['y']
        lat_nm = math.sqrt(dx * dx + dy * dy) * nm_per_pixel
        if lat_nm < min_sep:
            min_sep = lat_nm
            min_sep_t = t
    return (min_sep < PLANNING_LATERAL_NM, min_sep, min_sep_t)


def _search_clean_assignment(candidates, loc_locked, nm_per_pixel,
                             plan_steps, pair_cache):
    cs_order = sorted(candidates.keys())

    def pair_conflicts(cs_a, idx_a, cs_b, idx_b):
        ka, kb = (cs_a, idx_a), (cs_b, idx_b)
        if ka > kb:
            ka, kb = kb, ka
        key = (ka, kb)
        if key not in pair_cache:
            pair_cache[key] = _pair_conflict(
                candidates[ka[0]][ka[1]], candidates[kb[0]][kb[1]],
                nm_per_pixel, plan_steps)
        return pair_cache[key][0]

    chosen: dict = {}

    def backtrack(i):
        if i == len(cs_order):
            return True
        cs = cs_order[i]
        if cs not in candidates or not candidates[cs]:
            return False
        idx_range = [0] if cs in loc_locked else range(len(candidates[cs]))
        for idx in idx_range:
            ok = True
            for prev_cs in cs_order[:i]:
                if pair_conflicts(cs, idx, prev_cs, chosen[prev_cs]):
                    ok = False
                    break
            if not ok:
                continue
            chosen[cs] = idx
            if backtrack(i + 1):
                return True
        chosen.pop(cs, None)
        return False

    if backtrack(0):
        return dict(chosen)
    return None


def _min_sep_against(my_states, my_cs, all_plans, nm_per_pixel):
    ms = float('inf')
    for cs_other, p_other in all_plans.items():
        if cs_other == my_cs:
            continue
        other_states = _aligned_tail(p_other)
        n = min(len(my_states), len(other_states))
        for t in range(n):
            sa, sb = my_states[t], other_states[t]
            if bool(sa.get('on_ground')) != bool(sb.get('on_ground')):
                continue
            if abs(sa['alt'] - sb['alt']) >= WARN_VERTICAL_FT:
                continue
            dx = sa['x'] - sb['x']
            dy = sa['y'] - sb['y']
            lat_nm = math.sqrt(dx * dx + dy * dy) * nm_per_pixel
            if lat_nm < ms:
                ms = lat_nm
    return ms


def _maximize_separation(plans, all_attempts, loc_locked, nm_per_pixel, max_rounds=5):
    selectable = [cs for cs in all_attempts
                  if cs not in loc_locked and len(all_attempts[cs]) > 1]
    if not selectable:
        return plans
    chosen = dict(plans)
    for _outer in range(max_rounds):
        changed = False
        for cs in selectable:
            best_score = -1.0
            best_plan = None
            for candidate in all_attempts[cs]:
                tail = _aligned_tail(candidate)
                score = _min_sep_against(tail, cs, chosen, nm_per_pixel)
                if score > best_score:
                    best_score = score
                    best_plan = candidate
            if best_plan is None:
                continue
            if chosen.get(cs) is not best_plan:
                chosen[cs] = best_plan
                changed = True
        if not changed:
            break
    return chosen


def _state_to_row(state, callsign, sim_live):
    nmpp = sim_live.nm_per_pixel
    ax = sim_live.airport_x
    ay = sim_live.airport_y
    return {
        'callsign': callsign,
        'x_nm': str((state['x'] - ax) * nmpp),
        'y_nm': str(-(state['y'] - ay) * nmpp),
        'altitude': str(state['alt']),
        'heading': str(state['hdg']),
        'airspeed': str(state['spd']),
        'target_altitude': str(state.get('target_alt', state['alt'])),
        'target_heading': str(state.get('target_hdg', state['hdg'])),
        'target_airspeed': str(state.get('target_spd', state['spd'])),
        'loc': 'true' if state.get('loc') else 'false',
        'gs': 'true' if state.get('gs') else 'false',
        'on_ground': str(state.get('on_ground') or ''),
        'ils_runway': state.get('ils_runway') or '',
        'star': '', 'target_wpt': '', 'terminal': '',
    }


def _extend_plans_to_horizon(existing, plan_steps, sim_live, runtime, t0_sim,
                             airport, radar_side, suffix_counter):
    extended = dict(existing)
    work = []
    for cs, p in existing.items():
        if not p.states:
            continue
        last = p.states[-1]
        if last.get('landed') or last.get('on_ground'):
            continue
        # Imminent-touchdown guard: let live physics finish the landing.
        if (last.get('alt', float('inf')) < 100 and last.get('loc')
                and last.get('ils_runway')):
            continue
        remaining = p.remaining
        if remaining >= plan_steps:
            continue
        additional = plan_steps - remaining
        k = suffix_counter.get(cs, 0)
        suffix_counter[cs] = k + 1
        rollout_cs = f"{cs}_EXT_{k}"
        row = _state_to_row(last, cs, sim_live)
        ext_t0 = t0_sim + remaining
        work.append((cs, p, rollout_cs, row, ext_t0, additional))

    if not work:
        return extended

    pool = _ro.get_pool()
    if pool is not None:
        tasks = [(dict(row), rollout_cs, ext_t0, additional, airport, radar_side)
                 for (_cs, _p, rollout_cs, row, ext_t0, additional) in work]
        results = pool.map(_worker_rollout_task, tasks)
    else:
        results = []
        for (_cs, _p, rollout_cs, row, ext_t0, additional) in work:
            sub_sim, ac_obj = _build_subsim_with_plane(
                row, rollout_cs, ext_t0, airport, radar_side)
            if sub_sim is None:
                results.append(None)
                continue
            res = _run_rollout(sub_sim, ac_obj, rollout_cs, runtime, additional)
            try:
                runtime.forget(rollout_cs)
            except Exception:
                pass
            results.append(res)

    for (cs, original_plan, _, _, _, _), res in zip(work, results):
        if res is None:
            continue
        new_points, outcome = res
        if new_points is None:
            continue
        base_t = original_plan.states[-1]['t']
        appended = [dict(s, t=base_t + i)
                    for i, s in enumerate(new_points[1:], start=1)]
        extended[cs] = FlightPlan(
            callsign=cs, t0_sim=original_plan.t0_sim,
            states=original_plan.states + appended,
            cursor=original_plan.cursor, outcome=outcome,
            attempt=original_plan.attempt)
    return extended


def replan_all(sim_live, runtime, active_planes, existing_plans=None,
               plan_steps=400, max_conflict_iters=3, batch_size=3,
               airport='test', radar_side=800):
    """Build conflict-free flight plans for each active plane. Returns
    `(plans, residual_conflicts)`."""
    armed_set = {cs for cs in active_planes if cs in sim_live.aircraft_list}
    if not armed_set:
        return {}, []

    t0 = float(sim_live.sim_time)
    nmpp = sim_live.nm_per_pixel

    existing = {}
    if existing_plans:
        for cs, p in existing_plans.items():
            if cs in armed_set and not p.depleted:
                existing[cs] = p

    candidates: dict = {cs: [p] for cs, p in existing.items()}
    plans: dict = dict(existing)
    pair_cache: dict = {}

    rows: dict = {}

    def get_row(cs):
        if cs not in rows:
            rows[cs] = _snapshot_row(sim_live, sim_live.aircraft_list[cs], cs)
        return rows[cs]

    loc_locked = {cs for cs in armed_set
                  if sim_live.aircraft_list[cs].loc_intercepted}
    suffix_counter: dict = {cs: 0 for cs in armed_set}
    pending = {cs for cs in armed_set if cs not in plans}

    # Extension pass: roll preserved plans forward so their remaining tail
    # is at least plan_steps.
    if existing:
        existing = _extend_plans_to_horizon(
            existing, plan_steps, sim_live, runtime, t0, airport, radar_side,
            suffix_counter)
        plans = dict(existing)
        candidates = {cs: [p] for cs, p in existing.items()}

    def _accept(outcome):
        return outcome in ('LANDED', 'INTERMEDIATE')

    def _sample_batch(pending_set):
        if not pending_set:
            return
        pool = _ro.get_pool()
        if pool is not None:
            tasks, owners = [], []
            for cs in pending_set:
                for _ in range(batch_size):
                    k = suffix_counter[cs]
                    suffix_counter[cs] = k + 1
                    tasks.append((dict(get_row(cs)), f"{cs}_PLAN_{k}", t0,
                                  plan_steps, airport, radar_side))
                    owners.append((cs, k))
            if not tasks:
                return
            results = pool.map(_worker_rollout_task, tasks)
            for (cs, k), res in zip(owners, results):
                if res is None:
                    continue
                points, outcome = res
                if points is None or not _accept(outcome):
                    continue
                candidates.setdefault(cs, []).append(FlightPlan(
                    callsign=cs, t0_sim=t0, states=points, outcome=outcome, attempt=k))
        else:
            for cs in pending_set:
                for _ in range(batch_size):
                    k = suffix_counter[cs]
                    suffix_counter[cs] = k + 1
                    rollout_cs = f"{cs}_PLAN_{k}"
                    sub_sim, ac_obj = _build_subsim_with_plane(
                        get_row(cs), rollout_cs, t0, airport, radar_side)
                    if sub_sim is None:
                        continue
                    points, outcome = _run_rollout(
                        sub_sim, ac_obj, rollout_cs, runtime, plan_steps)
                    try:
                        runtime.forget(rollout_cs)
                    except Exception:
                        pass
                    if points is not None and _accept(outcome):
                        candidates.setdefault(cs, []).append(FlightPlan(
                            callsign=cs, t0_sim=t0, states=points,
                            outcome=outcome, attempt=k))

    for _iter_i in range(max_conflict_iters):
        if not pending:
            break
        _sample_batch(pending)

        for cs in candidates:
            if cs not in plans and candidates[cs]:
                plans[cs] = candidates[cs][0]

        if all(cs in candidates and candidates[cs] for cs in armed_set):
            assignment = _search_clean_assignment(
                candidates, loc_locked, nmpp, plan_steps, pair_cache)
            if assignment is not None:
                return {cs: candidates[cs][idx]
                        for cs, idx in assignment.items()}, []

        plans = _maximize_separation(plans, candidates, loc_locked, nmpp)
        tails = {cs: _aligned_tail(p) for cs, p in plans.items()}
        conflicts = find_conflicts(tails, nmpp, plan_steps)
        missing = armed_set - set(plans.keys())
        offenders = set()
        for c in conflicts:
            if c['a'] not in loc_locked:
                offenders.add(c['a'])
            if c['b'] not in loc_locked:
                offenders.add(c['b'])
        next_pending = missing | offenders
        if not next_pending:
            return plans, conflicts
        pending = next_pending

    plans = _maximize_separation(plans, candidates, loc_locked, nmpp)
    tails = {cs: _aligned_tail(p) for cs, p in plans.items()}
    return plans, find_conflicts(tails, nmpp, plan_steps)
