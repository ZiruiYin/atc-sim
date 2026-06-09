"""Multi-plane flight planning with conflict resolution.

Given a snapshot of the live sim, build a 200-step flight plan for each
active plane (armed + airborne). Plans are independent per-plane
rollouts; we then simulate the plans together and re-roll any plane
involved in a separation violation. Re-rolls are skipped for planes
already on LOC (their behavior is deterministic — we accept it).

The flight plan is a per-tick list of full aircraft states. During
playback the watch directly forces these on the live aircraft each tick
(no model query needed), which is fast and faithful — same target_*,
same x/y/alt/hdg, same loc/gs flags as the rollout produced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from heuristics_multiple.rollout import (
    _snapshot_row, _worker_rollout_task, _build_subsim_with_plane,
    _run_rollout, _pool,
)


# --------------------------------------------------------------------------- #
# Planning-stage conflict thresholds — unified with the watch's display
# rule (2 NM lateral / 1000 ft vertical, same-medium gated). A plan is
# "clean" iff it wouldn't trigger a warning when played back.
# --------------------------------------------------------------------------- #


PLANNING_LATERAL_NM = 2.0
WARN_VERTICAL_FT = 1000.0


@dataclass
class FlightPlan:
    """200-step recipe for one plane.

    `states[i]` is the full aircraft state at the rollout's i-th tick:
        {t, x, y, alt, hdg, spd, target_*, loc, gs, on_ground,
         ils_runway, landed}
    """
    callsign: str
    t0_sim: float
    states: list
    cursor: int = 0
    outcome: str = ''   # 'LANDED' | 'INTERMEDIATE'
    attempt: int = 0    # which rollout suffix produced this plan

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
# Conflict detection — pure-Python over the recorded per-tick state lists.
# Same-medium gating mirrors the strict rule on `watch.py:_check_aircraft_pair_strict`.
# --------------------------------------------------------------------------- #


def find_conflicts(plans: dict, nm_per_pixel: float,
                   max_ticks: int) -> list:
    """Return per-pair conflict records under the 2 NM / 1000 ft
    same-medium rule — unified with the live display rule (see
    PLANNING_LATERAL_NM at the top of this module).

    Each record is a dict:
        {'a', 'b': callsigns (a < b in cs_list order),
         'first_t': first tick the pair entered the conflict cone,
         'min_sep_nm': closest lateral approach over all ticks,
         'min_sep_t': tick at which min separation occurred}

    A pair appears in the list iff it ever entered the conflict cone
    (lateral < 2 NM AND vertical < 1000 ft AND same medium) at any
    tick within [0, max_ticks].
    """
    cs_list = list(plans.keys())
    pair_data: dict = {}
    for t in range(max_ticks + 1):
        for i in range(len(cs_list)):
            for j in range(i + 1, len(cs_list)):
                a, b = cs_list[i], cs_list[j]
                pair = (a, b)
                ta = plans[a]
                tb = plans[b]
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
                rec = pair_data.get(pair)
                if rec is None:
                    pair_data[pair] = {
                        'a': a, 'b': b,
                        'first_t': t,
                        'min_sep_nm': lateral_nm,
                        'min_sep_t': t,
                    }
                elif lateral_nm < rec['min_sep_nm']:
                    rec['min_sep_nm'] = lateral_nm
                    rec['min_sep_t'] = t
    return list(pair_data.values())


# --------------------------------------------------------------------------- #
# Single-plane rollout helper — uses the worker pool when available,
# else falls back to in-process execution.
# --------------------------------------------------------------------------- #


def _roll_one(sim_live, runtime, cs, row, t0, plan_steps,
              airport, radar_side, attempt):
    """Run one rollout in-process. Returns (states, outcome) or
    (None, failure_reason)."""
    rollout_cs = f"{cs}_PLAN_{attempt}"
    sub_sim, ac_obj = _build_subsim_with_plane(
        row, rollout_cs, t0, airport, radar_side)
    if sub_sim is None:
        try:
            runtime.forget(rollout_cs)
        except Exception:
            pass
        return None, 'NO_SUB_SIM'
    result = _run_rollout(sub_sim, ac_obj, rollout_cs, runtime, plan_steps)
    try:
        runtime.forget(rollout_cs)
    except Exception:
        pass
    return result


def _roll_batch_parallel(pool, rows, planes, suffix_counter,
                         t0, plan_steps, airport, radar_side):
    """Dispatch one rollout per plane to the worker pool. Returns
    {cs: (points, outcome)} for planes whose rollout returned non-None
    (keeps both LANDED and INTERMEDIATE). Failed planes are absent."""
    tasks = []
    owners = []
    for cs in planes:
        k = suffix_counter[cs]
        suffix_counter[cs] = k + 1
        rollout_cs = f"{cs}_PLAN_{k}"
        tasks.append((dict(rows[cs]), rollout_cs, t0, plan_steps,
                      airport, radar_side))
        owners.append((cs, k))
    if not tasks:
        return {}
    results = pool.map(_worker_rollout_task, tasks)
    out = {}
    for (cs, k), res in zip(owners, results):
        if res is None:
            continue
        points, outcome = res
        if points is None:
            continue
        out[cs] = (points, outcome, k)
    return out


# --------------------------------------------------------------------------- #
# Top-level: build plans for all active planes with conflict resolution.
# --------------------------------------------------------------------------- #


def _aligned_tail(plan: 'FlightPlan') -> list:
    """States from the plan's cursor onward — i.e. states[cursor:].

    Every plan's tail starts at the SAME absolute sim time (the moment
    the replanning call was made), so two tails can be index-aligned
    for conflict detection regardless of when each plan was originally
    rolled.
    """
    return plan.states[plan.cursor:]


def _pair_conflict(plan_a: 'FlightPlan', plan_b: 'FlightPlan',
                   nm_per_pixel: float, max_ticks: int) -> tuple:
    """Walk two plans' aligned tails tick-by-tick and return
    `(conflicting, min_sep_nm, min_sep_t)`. A pair "conflicts" iff
    their min separation falls inside the 2 NM lateral / 1000 ft
    vertical / same-medium cone at any tick.
    """
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


def _search_clean_assignment(candidates: dict, loc_locked: set,
                              nm_per_pixel: float, plan_steps: int,
                              pair_cache: dict) -> 'dict | None':
    """Backtracking constraint search.

    For each plane, pick one candidate index such that no pair of chosen
    candidates conflicts. LOC-locked planes are forced to index 0 (their
    existing plan, which we never re-roll).

    `pair_cache` is keyed by `((cs_a, idx_a), (cs_b, idx_b))` with
    `(cs_a, idx_a) <= (cs_b, idx_b)` lexicographically. It carries
    pair-conflict results across iterations of replan_all so we never
    re-evaluate the same pair twice (the "pruning" the user asked for).

    Returns `{cs: chosen_idx}` if a clean assignment exists, else None.
    """
    cs_order = sorted(candidates.keys())

    def pair_conflicts(cs_a, idx_a, cs_b, idx_b):
        ka, kb = (cs_a, idx_a), (cs_b, idx_b)
        if ka > kb:
            ka, kb = kb, ka
        key = (ka, kb)
        if key not in pair_cache:
            pair_cache[key] = _pair_conflict(
                candidates[ka[0]][ka[1]],
                candidates[kb[0]][kb[1]],
                nm_per_pixel, plan_steps)
        return pair_cache[key][0]

    chosen: dict = {}

    def backtrack(i: int) -> bool:
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


def _min_sep_against(my_states: list, my_cs: str,
                     all_plans: dict, nm_per_pixel: float) -> float:
    """Minimum lateral separation (NM) of `my_states` against every
    other plane's *aligned tail* in `all_plans`. Same-medium gated;
    only counted within the 1000 ft vertical bin. Returns +inf if
    `my_states` never coexists at altitude with anyone.

    `my_states` is assumed to already start at current sim time (either
    a freshly-rolled plan's states[0:] or an existing plan's tail).
    """
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


def _maximize_separation(plans: dict, all_attempts: dict,
                         loc_locked: set, nm_per_pixel: float,
                         max_rounds: int = 5) -> dict:
    """Greedy fallback when find_conflicts can't be driven to zero:
    for each non-LOC plane that has multiple plan candidates on record,
    swap its plan to the one whose aligned tail maximizes min-lateral
    separation against the currently-chosen plans of the other planes.
    Iterate until no further swaps help (or `max_rounds` reached).

    Coordinate-ascent on a discrete grid — not a global optimum but a
    strict improvement over the last sampled set.

    `all_attempts[cs]` is a list of `FlightPlan` candidates. Each plan's
    aligned tail (`states[cursor:]`) is what gets scored — so existing
    plans whose cursor has advanced are compared on equal terms with
    freshly-rolled candidates.
    """
    selectable = [cs for cs in all_attempts
                  if cs not in loc_locked and len(all_attempts[cs]) > 1]
    if not selectable:
        return plans

    chosen = dict(plans)
    for outer in range(max_rounds):
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
            cur = chosen.get(cs)
            if cur is not best_plan:
                chosen[cs] = best_plan
                changed = True
        if not changed:
            break
    return chosen


def _state_to_row(state: dict, callsign: str, sim_live) -> dict:
    """Reverse the (px, py) -> (x_nm, y_nm) projection inside
    `_snapshot_row` so we can reconstruct an aircraft from a recorded
    flight-plan state dict. Used by the extension pass to re-spawn the
    plane in a sub_sim at the end of its existing plan."""
    nmpp = sim_live.nm_per_pixel
    ax = sim_live.airport_x
    ay = sim_live.airport_y
    x_nm = (state['x'] - ax) * nmpp
    y_nm = -(state['y'] - ay) * nmpp
    return {
        'callsign': callsign,
        'x_nm': str(x_nm),
        'y_nm': str(y_nm),
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
        'star': '',
        'target_wpt': '',
        'terminal': '',
    }


def _extend_plans_to_horizon(existing: dict, plan_steps: int,
                              sim_live, runtime, t0_sim: float,
                              airport: str, radar_side: int,
                              suffix_counter: dict) -> dict:
    """For every existing plan whose remaining states are fewer than
    `plan_steps`, roll out the gap from the plan's last state and
    append. Returns a new dict of FlightPlans.

    Skipped:
      - depleted plans (caller already excluded these),
      - plans that already end with a landed state (plane touches down
        within its existing tail; extension is meaningless).

    Extension is dispatched to the worker pool when available — each
    plane's extension is an independent rollout, identical in shape to
    a fresh planning rollout.
    """
    extended = dict(existing)
    work = []
    for cs, p in existing.items():
        if not p.states:
            continue
        last = p.states[-1]
        if last.get('landed') or last.get('on_ground'):
            continue
        # Imminent-touchdown guard: at very low altitude on LOC with
        # ils_runway set, the plane is seconds away from update_landed
        # firing in the LIVE sim once this plan depletes. Extending
        # from here re-rolls a sub_sim whose snapshot caught the
        # touched-down ac mid-decel (target=140 from the LOC override,
        # airspeed bleeding toward 0); the resulting trajectory wedges
        # the live plane at low alt with airspeed=0, ils_runway still
        # set, and never resolves — visible as a "stuck on runway"
        # bug. Just let live physics finish the touchdown.
        if (last.get('alt', float('inf')) < 100
                and last.get('loc')
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

    if _pool is not None:
        tasks = [(dict(row), rollout_cs, ext_t0, additional,
                  airport, radar_side)
                 for (_cs, _p, rollout_cs, row, ext_t0, additional) in work]
        results = _pool.map(_worker_rollout_task, tasks)
    else:
        results = []
        for (_cs, _p, rollout_cs, row, ext_t0, additional) in work:
            sub_sim, ac_obj = _build_subsim_with_plane(
                row, rollout_cs, ext_t0, airport, radar_side)
            if sub_sim is None:
                results.append(None)
                continue
            res = _run_rollout(sub_sim, ac_obj, rollout_cs,
                               runtime, additional)
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
        appended = []
        for i, np in enumerate(new_points[1:], start=1):
            s = dict(np)
            s['t'] = base_t + i
            appended.append(s)
        extended[cs] = FlightPlan(
            callsign=cs,
            t0_sim=original_plan.t0_sim,
            states=original_plan.states + appended,
            cursor=original_plan.cursor,
            outcome=outcome,
            attempt=original_plan.attempt,
        )
    return extended


def replan_all(sim_live, runtime, active_planes,
               existing_plans: 'dict | None' = None,
               plan_steps: int = 500,
               max_conflict_iters: int = 3,
               batch_size: int = 3,
               airport: str = 'test',
               radar_side: int = 800,
               full_rollouts: bool = False,
               full_rollout_max_steps: int = 5000) -> tuple:
    """Build conflict-free flight plans for each active plane.

    Returns `(plans, residual_conflicts)`. `plans` is `{cs: FlightPlan}`
    for every plane that ended up with a valid rollout.

    ALGORITHM (batched sample + search):
      1. Preserve any provided `existing_plans` as candidate #0 for each
         plane (cursor preserved). LOC-locked planes are FIXED to #0 —
         never re-rolled.
      2. Each iteration: for every plane currently in `pending`, sample
         `batch_size` (default 3) fresh rollouts in parallel and append
         to that plane's candidate pool.
      3. Run a backtracking search across the accumulated pool to find
         an assignment (one candidate per plane) with no pair conflicts
         under the 2 NM / 1000 ft same-medium rule. Pair-conflict checks
         are memoised across iterations — once a (cs_a, idx_a, cs_b, idx_b)
         pair is known to conflict (or not), we never re-evaluate.
      4. If the search succeeds, return that assignment immediately.
      5. Otherwise, greedy-pick the best partial assignment from the
         current pool, mark non-LOC offenders + missing planes as the
         next round's `pending`, and loop.
      6. After `max_conflict_iters`, fall back to the max-separation
         greedy across the full candidate pool and report residuals.

    Per-plane pool growth: 3, 6, 9, ... up to 3 * 9 = 27 candidates
    (plus the existing plan if present) for a plane that's in conflict
    every round.
    """
    armed_set = {cs for cs in active_planes if cs in sim_live.aircraft_list}
    if not armed_set:
        return {}, []

    t0 = float(sim_live.sim_time)
    nmpp = sim_live.nm_per_pixel

    # Preserve existing plans for active planes whose plans aren't depleted.
    existing = {}
    if existing_plans:
        for cs, p in existing_plans.items():
            if cs in armed_set and not p.depleted:
                existing[cs] = p

    # Per-plane candidate pool. Existing plan (if any) is candidate #0.
    candidates: dict = {cs: [p] for cs, p in existing.items()}
    # Current chosen plans: initialised from existing.
    plans: dict = dict(existing)

    # Persistent pair-conflict memo across all iterations of THIS call.
    pair_cache: dict = {}

    rows: dict = {}
    def get_row(cs):
        if cs not in rows:
            rows[cs] = _snapshot_row(sim_live, sim_live.aircraft_list[cs], cs)
        return rows[cs]

    loc_locked = {cs for cs in armed_set
                  if sim_live.aircraft_list[cs].loc_intercepted}
    suffix_counter: dict = {cs: 0 for cs in armed_set}

    # Initial pending: planes without a preserved plan.
    pending = {cs for cs in armed_set if cs not in plans}

    # --- Extension pass --------------------------------------------------
    # Preserved existing plans get rolled forward from their last state
    # so their REMAINING tail is at least `plan_steps`. Skipped in
    # full-rollouts mode (plans already end at landing — extending is
    # meaningless and would re-fail). Also skipped when there's nothing
    # to preserve.
    if existing and not full_rollouts:
        existing = _extend_plans_to_horizon(
            existing, plan_steps, sim_live, runtime,
            t0, airport, radar_side, suffix_counter)
        plans = dict(existing)
        candidates = {cs: [p] for cs, p in existing.items()}

    # In full-rollouts mode, each rollout runs until LANDED (or fails
    # for another reason); cap at full_rollout_max_steps as a safety.
    rollout_max_steps = (full_rollout_max_steps if full_rollouts
                          else plan_steps)

    def _accept_outcome(outcome: str) -> bool:
        # Full mode: only LANDED rollouts are valid candidates.
        # Standard mode: INTERMEDIATE (hit max_steps without explicit
        # failure) is also kept.
        if full_rollouts:
            return outcome == 'LANDED'
        return outcome in ('LANDED', 'INTERMEDIATE')

    def _sample_batch(pending_set):
        """Sample `batch_size` fresh candidates for every plane in
        `pending_set` in parallel and append to `candidates`."""
        if not pending_set:
            return
        if _pool is not None:
            tasks = []
            owners = []
            for cs in pending_set:
                for _ in range(batch_size):
                    k = suffix_counter[cs]
                    suffix_counter[cs] = k + 1
                    rollout_cs = f"{cs}_PLAN_{k}"
                    tasks.append((dict(get_row(cs)), rollout_cs, t0,
                                  rollout_max_steps, airport, radar_side))
                    owners.append((cs, k))
            if not tasks:
                return
            results = _pool.map(_worker_rollout_task, tasks)
            for (cs, k), res in zip(owners, results):
                if res is None:
                    continue
                points, outcome = res
                if points is None:
                    continue
                if not _accept_outcome(outcome):
                    continue
                new_plan = FlightPlan(
                    callsign=cs, t0_sim=t0,
                    states=points, outcome=outcome, attempt=k)
                candidates.setdefault(cs, []).append(new_plan)
        else:
            for cs in pending_set:
                for _ in range(batch_size):
                    k = suffix_counter[cs]
                    suffix_counter[cs] = k + 1
                    points, outcome = _roll_one(
                        sim_live, runtime, cs, get_row(cs), t0,
                        rollout_max_steps, airport, radar_side, k)
                    if points is not None and _accept_outcome(outcome):
                        new_plan = FlightPlan(
                            callsign=cs, t0_sim=t0,
                            states=points, outcome=outcome, attempt=k)
                        candidates.setdefault(cs, []).append(new_plan)

    for iter_i in range(max_conflict_iters):
        if not pending:
            break

        _sample_batch(pending)

        # Initialise plans for newly-seen planes (so the greedy fallback
        # has something to work with even if the clean-search fails).
        for cs in candidates:
            if cs not in plans and candidates[cs]:
                plans[cs] = candidates[cs][0]

        # Only run the clean-assignment search once every plane has at
        # least one candidate — otherwise a plane with all-failed rolls
        # would block the search.
        if all(cs in candidates and candidates[cs] for cs in armed_set):
            assignment = _search_clean_assignment(
                candidates, loc_locked, nmpp, rollout_max_steps, pair_cache)
            if assignment is not None:
                chosen = {cs: candidates[cs][idx]
                          for cs, idx in assignment.items()}
                return chosen, []

        # No clean assignment yet — greedy-pick the best from current
        # pool to use as the "current state" for offender identification.
        plans = _maximize_separation(plans, candidates, loc_locked, nmpp)

        tails = {cs: _aligned_tail(p) for cs, p in plans.items()}
        conflicts = find_conflicts(tails, nmpp, rollout_max_steps)
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

    # Out of iterations — final greedy max-sep pick across the entire
    # candidate pool, then report whatever residual conflicts remain.
    plans = _maximize_separation(plans, candidates, loc_locked, nmpp)
    tails = {cs: _aligned_tail(p) for cs, p in plans.items()}
    return plans, find_conflicts(tails, nmpp, rollout_max_steps)
