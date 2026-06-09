"""Fair multi-plane eval — model-driven, single-sim, TIMEOUT-enforcing.

Mirrors `heuristics_multiple/eval.py` methodology EXACTLY (single
`SimulationEnv` per scenario, restart only on natural crash, pre-LOC
step cap producing TIMEOUT outcomes, strict 2 NM / 1000 ft / same-medium
collision rule, same CSV format) — but uses a `Runtime` / `MultiRuntime`
to drive armed planes instead of flight-plan replanning.

Per scenario:
  1. Spin up a fresh `SimulationEnv` with the strict collision rule.
  2. Arm planes after `warmup_wpts` STAR waypoints; from then on the
     runtime ticks them.
  3. Apply the pre-LOC step cap — planes armed for > `max_steps_1_2`
     (STAR1/2) or > `max_steps_3` (STAR3) without intercepting LOC are
     force-removed and recorded as TIMEOUT.
  4. Stop on `sim.crash_occurred` or after `max_steps` ticks (`max_steps`
     is a safety belt at 1_000_000 — never fires in practice).

Outcomes (per plane):
  LANDED        — touched down on runway
  CRASHED       — terminating collision pair
  IMPROPER_EXIT — flew out of radar bounds
  TIMEOUT       — pre-LOC step cap fired
  TRUNCATED    — sim ended before this plane finished (dropped from metrics)

Metrics:
  kept            = LANDED + CRASHED + IMPROPER_EXIT + TIMEOUT
  policy_sr       = LANDED / kept
  num_crashes     = absolute count of CRASHED
  avg_violation_s = mean per-plane violation seconds (collision_warning ticks)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.core.simulation import SimulationEnv  # noqa: E402
from environment.core.human_data_logger import HumanDataRecorder  # noqa: E402

# Reuse helpers from the heuristics eval — identical methodology.
from heuristics_multiple.eval import (  # noqa: E402
    Traj,
    ScenarioResult,
    _seed_all,
    _arm_plane,
    _check_arm,
    _enforce_pre_loc_timeout,
    _install_strict_collision_rule,
    compute_metrics,
    keep_outcome,
)


def run_one_scenario(runtime,
                     seed: int,
                     spawn_rate: int = 90,
                     max_steps: int = 1_000_000,
                     max_steps_1_2: int = 1200,
                     max_steps_3: int = 500,
                     airport: str = 'test',
                     runway: str = '27',
                     warmup_wpts: int = 2,
                     out_csv: Optional[Path] = None,
                     stop_after_kept: Optional[int] = None,
                     ) -> ScenarioResult:
    """Run one model-driven scenario. Returns trajectories + crash time.

    Identical to `heuristics_multiple.eval.run_one_scenario` minus the
    flight-plan machinery — the runtime ticks all armed planes directly.
    """
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

    plane_spawn_t: dict = {}
    plane_violation_s: dict = {}
    completions: list = []
    completed_cs: set = set()
    crashed_callsigns: set = set()
    armed: set = set()
    armed_at: dict = {}
    plane_star: dict = {}
    pre_loc_steps: dict = {}
    initial_star_len: dict = {}

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
            flight_plans={},  # model-driven: no flight plans to clean up
            armed_at=armed_at,
            initial_star_len=initial_star_len,
            plane_spawn_t=plane_spawn_t,
            plane_violation_s=plane_violation_s,
            completions=completions,
            completed_cs=completed_cs,
            runtime=runtime,
        )

        # Model drives every armed plane in one shot.
        runtime.tick(sim, armed=armed)
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

        if stop_after_kept is not None:
            kept_count = sum(1 for t in completions
                             if t.outcome not in ('TRUNCATED', 'UNKNOWN'))
            if kept_count >= stop_after_kept:
                break

        # Attribute removals from this sim.step (TIMEOUT removals excluded
        # via completed_cs).
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
            armed.discard(cs)
            armed_at.pop(cs, None)
            initial_star_len.pop(cs, None)
            plane_star.pop(cs, None)
            pre_loc_steps.pop(cs, None)

    # Anything still in the air at scenario end = TRUNCATED.
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


__all__ = [
    'run_one_scenario',
    'Traj',
    'ScenarioResult',
    'compute_metrics',
    'keep_outcome',
]
