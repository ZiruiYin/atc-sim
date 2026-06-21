"""AUTO planner — drives every SIMULATED-airport aircraft with the GMM policy
plus multi-plane conflict resolution.

Matches the rl `heuristics_multiple` **watch** (live) loop, not the synchronous
eval harness:

  - **Planning runs in a background thread.** While a replan is in flight the
    sim HOLDS (does not advance) and the UI shows a PLANNING overlay. This keeps
    `/step`, `/auto off`, and `/restart` responsive — they never block on a
    multi-second replan.
  - **Arming:** a plane is taken over once it has flown `warmup_wpts` of its
    STAR (measured from the FULL procedure length, so a plane already past
    warmup when AUTO is switched on mid-scenario is armed immediately). On
    arming its STAR is cleared and it is issued `L <runway>`.
  - **Replan trigger:** a newly-armed active plane with no plan, or every active
    plan depleted. Existing plans are extended to the horizon; conflicts are
    re-rolled up to `max_conflict_iters`. Horizon = `plan_steps` (default 400).
  - **Live application:** each tick the planned state is forced onto every
    planned aircraft (teleport replay) and the GMM drives any armed plane that
    has no plan yet.
  - **Termination is the simulator's:** crash / landed / improper-exit. No
    timeouts; no LOC-above-GS dropping.
  - **AUTO off:** physics is restored and every armed plane that is not already
    on the localizer is left **cleared to land** (`L <runway>`) so it keeps
    flying the approach instead of freezing on its last vector.

CPU: rollouts run on an adaptive spawn pool sized to `cpu_count - 1`
(serial on a 1-2 core host).
"""

from __future__ import annotations

import os
import threading
import traceback
import types

from environment.params import TRAJ_LENGTH

from auto_plan import rollout as _ro
from auto_plan.flight_plan import replan_all, FlightPlan  # noqa: F401


def _adaptive_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


class AutoPlanner:
    def __init__(self, airport: str = 'test', runway: str = '27',
                 plan_steps: int = 400, warmup_wpts: int = 2,
                 batch_size: int = 3, max_conflict_iters: int = 3):
        self.airport = airport
        self.runway = runway
        self.plan_steps = int(plan_steps)
        self.warmup_wpts = int(warmup_wpts)
        self.batch_size = int(batch_size)
        self.max_conflict_iters = int(max_conflict_iters)

        self.runtime = None
        self.n_workers = 1
        self.started = False

        self.armed: set = set()
        self.initial_star_len: dict = {}
        self.flight_plans: dict = {}

        self._lock = threading.Lock()
        self._planning = False
        self._planning_thread = None
        self._plan_gen = 0            # bumped on reset/enable to discard stale replans
        self._residual_conflicts: list = []
        self._unplanned: set = set()

    # ---------------- lifecycle ---------------- #

    def start(self) -> dict:
        """Load the policy and (adaptively) warm a worker pool. Idempotent —
        a no-op once started, so re-enabling AUTO is instant."""
        if self.started:
            return self.status()
        from auto_plan.runtime import Runtime
        self.runtime = Runtime(runway=self.runway)
        self.n_workers = _adaptive_workers()
        _ro.init_pool(self.runtime, self.n_workers, warm=True)   # None if <=1
        self.started = True
        return self.status()

    def enable(self, sim) -> None:
        """Called when AUTO is switched on (possibly mid-scenario). Clears any
        stale state; arming + the first replan happen on the next step()."""
        self.reset(sim)

    def disable(self, sim) -> None:
        """AUTO off: restore physics and leave every armed plane that is not on
        the localizer cleared to land, so they keep flying the approach."""
        # Bump the generation so any in-flight replan discards its result; let
        # that thread clear `_planning` itself (don't race its pool.map).
        with self._lock:
            self._plan_gen += 1
        self._restore_all(sim)
        for cs in list(self.armed):
            ac = sim.aircraft_list.get(cs)
            if ac is None:
                continue
            if (ac.loc_intercepted or ac.gs_intercepted
                    or ac.on_ground or ac.landed):
                continue   # already landing — don't touch
            ac.star = None
            ac.star_name = None
            ac.target_wpt = None
            sim.command(cs, f"L {self.runway}")
        self.flight_plans.clear()
        self.armed.clear()
        self.initial_star_len.clear()
        self._residual_conflicts = []
        self._unplanned = set()
        if self.runtime is not None:
            try:
                self.runtime.reset()
            except Exception:
                pass

    def reset(self, sim=None) -> None:
        """Clear per-scenario state (on restart / airport switch / enable) but
        keep the loaded policy + warm pool."""
        with self._lock:
            self._plan_gen += 1   # in-flight replan (if any) will self-discard
        if sim is not None:
            self._restore_all(sim)
        self.flight_plans.clear()
        self.armed.clear()
        self.initial_star_len.clear()
        self._residual_conflicts = []
        self._unplanned = set()
        if self.runtime is not None:
            try:
                self.runtime.reset()
            except Exception:
                pass

    def stop(self, sim=None) -> None:
        """Full teardown — also shuts the worker pool down."""
        self.reset(sim)
        _ro.shutdown_pool()
        self.started = False

    def status(self) -> dict:
        return {
            'on': self.started,
            'n_workers': self.n_workers,
            'mode': 'pool' if self.n_workers > 1 else 'serial',
            'plan_steps': self.plan_steps,
            'n_armed': len(self.armed),
            'n_planned': len(self.flight_plans),
            'planning': self._planning,
        }

    def overlay(self) -> dict:
        """State-payload additions for the UI: planning flag + remaining
        flight-plan tails (light-blue lines) + unresolved conflicts."""
        out = {'auto': self.started, 'planning': bool(self._planning)}
        fp = {}
        with self._lock:
            for cs, plan in self.flight_plans.items():
                tail = plan.states[plan.cursor:]
                if len(tail) < 2:
                    continue
                fp[cs] = {'states': [{'x': s['x'], 'y': s['y']} for s in tail]}
            conflicts = list(self._residual_conflicts)
        out['flight_plans'] = fp
        if conflicts:
            out['plan_conflicts'] = [{'a': c['a'], 'b': c['b']} for c in conflicts]
        return out

    # ---------------- per-step (mirrors watch /step plan-mode body) -------- #

    def step(self, sim) -> None:
        """Advance the sim under AUTO control. While a replan is in flight the
        sim HOLDS (no stepping). Call this instead of sim.step() when AUTO is
        on; it runs the full fast_forward loop itself."""
        if not self.started or self.runtime is None:
            for _ in range(sim.fast_forward):
                sim.step(1.0)
            return

        # Prune bookkeeping for planes the sim has removed. armed /
        # initial_star_len are only touched on this (main) thread.
        live = set(sim.aircraft_list.keys())
        self.armed &= live
        for cs in [c for c in self.initial_star_len if c not in live]:
            del self.initial_star_len[cs]

        if self._planning:
            return   # hold the sim while a background replan runs

        # flight_plans is owned exclusively by this thread when not planning.
        for cs in [c for c in self.flight_plans if c not in live]:
            self.flight_plans.pop(cs, None)

        self._check_arm(sim)
        active = self._active_set(sim)
        if self._replan_needed(active):
            self._trigger_replan_async(sim, active)
            return

        for _ in range(sim.fast_forward):
            self._check_arm(sim)
            active_now = self._active_set(sim)
            if self._replan_needed(active_now):
                self._trigger_replan_async(sim, active_now)
                break
            self._apply_flight_plans(sim)
            armed_no_plan = self.armed - set(self.flight_plans.keys())
            self.runtime.tick(sim, armed=armed_no_plan)
            sim.step(1.0)
        self._check_arm(sim)

    # ---------------- async replanning ---------------- #

    def _trigger_replan_async(self, sim, active) -> None:
        with self._lock:
            if self._planning:
                return
            self._planning = True
            gen = self._plan_gen
        active_local = set(active)
        existing_local = dict(self.flight_plans)

        def _do_replan():
            try:
                new_plans, residual = replan_all(
                    sim_live=sim, runtime=self.runtime,
                    active_planes=active_local, existing_plans=existing_local,
                    plan_steps=self.plan_steps,
                    max_conflict_iters=self.max_conflict_iters,
                    batch_size=self.batch_size, airport=self.airport,
                    radar_side=sim.radar_side)
                with self._lock:
                    if gen != self._plan_gen:
                        return   # superseded by a reset/disable — discard
                    # Un-no-op any aircraft from the prior plan set.
                    for cs, plan in self.flight_plans.items():
                        ac = sim.aircraft_list.get(cs)
                        if ac is not None and hasattr(ac, '_plan_orig_update'):
                            ac.update = ac._plan_orig_update
                            delattr(ac, '_plan_orig_update')
                    self.flight_plans = new_plans
                    self._residual_conflicts = residual
                    self._unplanned = active_local - set(new_plans.keys())
            except Exception:
                traceback.print_exc()
            finally:
                with self._lock:
                    self._planning = False

        self._planning_thread = threading.Thread(target=_do_replan, daemon=True)
        self._planning_thread.start()

    # ---------------- arming / active ---------------- #

    def _arm_plane(self, sim, cs, ac) -> None:
        if ac.star is not None or ac.target_wpt is not None:
            ac.star = None
            ac.star_name = None
            ac.target_wpt = None
        res = sim.command(cs, f"L {self.runway}")
        if res.get('ok'):
            self.runtime.state_for(cs).cleared = True

    def _check_arm(self, sim) -> None:
        procedures = sim.data.get('star_procedures', {})
        for cs, ac in sim.aircraft_list.items():
            if cs in self.armed:
                continue
            if cs not in self.initial_star_len:
                # Measure warmup from the FULL STAR length so a plane already
                # past warmup when AUTO turns on mid-scenario arms immediately.
                if ac.star_name and ac.star_name in procedures:
                    self.initial_star_len[cs] = len(procedures[ac.star_name])
                else:
                    self.initial_star_len[cs] = len(ac.star) if ac.star else 0
            initial = self.initial_star_len[cs]
            current = len(ac.star) if ac.star else 0
            popped = initial - current
            threshold = min(self.warmup_wpts, initial) if initial > 0 else 0
            if initial == 0 or popped >= threshold:
                self._arm_plane(sim, cs, ac)
                self.armed.add(cs)

    def _active_set(self, sim) -> set:
        return {cs for cs in self.armed
                if cs in sim.aircraft_list
                and not sim.aircraft_list[cs].on_ground}

    def _replan_needed(self, active) -> bool:
        if not active:
            return False
        if not self.flight_plans:
            return True
        for cs in active:
            if cs not in self.flight_plans:
                return True
        if all(self.flight_plans[cs].depleted for cs in active
               if cs in self.flight_plans):
            return True
        return False

    # ---------------- plan application ---------------- #

    @staticmethod
    def _restore_one(ac) -> None:
        if ac is not None and hasattr(ac, '_plan_orig_update'):
            ac.update = ac._plan_orig_update
            delattr(ac, '_plan_orig_update')

    def _restore_all(self, sim) -> None:
        for cs in list(self.flight_plans.keys()):
            self._restore_one(sim.aircraft_list.get(cs))

    def _apply_flight_plans(self, sim) -> None:
        if not self.flight_plans:
            return
        with self._lock:
            plan_items = list(self.flight_plans.items())
        for cs, plan in plan_items:
            ac = sim.aircraft_list.get(cs)
            if ac is None:
                self.flight_plans.pop(cs, None)
                continue
            if plan.depleted:
                self._restore_one(ac)
                self.flight_plans.pop(cs, None)
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
