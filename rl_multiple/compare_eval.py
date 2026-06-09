"""Paired multi-plane A/B eval — compare two policies on identical scenarios.

Workflow per seed:
  1. Build two SimulationEnv instances. Seed Python/Numpy/Torch RNG to `seed`
     before each so the spawner produces identical plane sequences for both.
  2. Run model A's runtime against sim A, ticking until crash or `max_steps`.
     Record per-plane (callsign, outcome, term_time, violation_seconds,
     spawn_time). Outcomes ∈ {LANDED, IMPROPER_EXIT, CRASHED, TRUNCATED}.
  3. Repeat for model B with seed reset.
  4. Fairness cap: t_cap = min(crash_time_A or ∞, crash_time_B or ∞, max_steps).
     Any traj that didn't terminate by t_cap is re-classified TRUNCATED and
     dropped. The crashed traj from each side stays as CRASHED (it terminated
     ON t_cap by definition for that side).
  5. Aggregate kept trajectories until each model has ≥ `n_target`.

Metrics:
  policy_sr      = LANDED / (LANDED + IMPROPER_EXIT)         (non-crashed kept)
  crash_rate     = CRASHED / total_kept
  avg_violation_s = mean(violation_seconds across kept trajs)

Per-plane violation_seconds = sim seconds during which that plane had
collision_warning=True (under the watch's strict 2-NM / 1000-ft rule).
"""

from __future__ import annotations

import json
import random
import sys
import time
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

from rl_multiple.runtime import Runtime, MultiRuntime  # noqa: E402


# --------------------------------------------------------------------------- #
# Strict pair-check rule (mirrors watch._check_aircraft_pair_strict).
# 2 NM lateral + 1000 ft vertical → collision_warning; the upstream
# hard-crash threshold (≤50 ft vert, ≤0.2 NM lat) is preserved verbatim.
# We monkey-patch the sim's CollisionMonitor instance — the default rule
# suppresses warnings whenever either plane is on-LOC/on-ground, which
# silently mutes the very failure mode we want to evaluate.
# --------------------------------------------------------------------------- #
WARN_LATERAL_NM = 2.0
WARN_VERTICAL_FT = 1000.0


def _check_aircraft_pair_strict(self, aircraft1, aircraft2):
    pixel_distance = distance_between_coords_pixels(
        aircraft1.x, aircraft1.y, aircraft2.x, aircraft2.y)
    lateral_nm = pixel_distance * self.nm_per_pixel
    vertical_separation = abs(aircraft1.altitude - aircraft2.altitude)

    if lateral_nm < WARN_LATERAL_NM and vertical_separation < WARN_VERTICAL_FT:
        aircraft1.collision_warning = True
        aircraft2.collision_warning = True

    if vertical_separation <= 50:
        crash_threshold_pixels = 0.2 / self.nm_per_pixel
        if pixel_distance <= crash_threshold_pixels:
            aircraft1.crash = f"collided with {aircraft2.callsign}"
            aircraft2.crash = f"collided with {aircraft1.callsign}"


def _install_strict_collision_rule(sim_obj: SimulationEnv) -> None:
    import types
    sim_obj.collision_monitor._check_aircraft_pair = types.MethodType(
        _check_aircraft_pair_strict, sim_obj.collision_monitor)


@dataclass
class Traj:
    callsign: str
    outcome: str           # LANDED | IMPROPER_EXIT | CRASHED | TRUNCATED
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
    trajs: list[Traj] = field(default_factory=list)
    csv_path: Optional[str] = None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _arm_plane(sim, runtime, cs, ac, runway: str) -> None:
    """Mirror watch._arm_ppo / MultiRolloutSim._arm_plane.

    Clears the plane's STAR/target_wpt, issues `LAND <runway>` so the sim
    sets ils_runway and the policy can run LOC/GS, and (for MultiRuntime)
    flips the per-callsign `cleared` flag.
    """
    if ac.star is not None or ac.target_wpt is not None:
        ac.star = None
        ac.star_name = None
        ac.target_wpt = None
    res = sim.command(cs, f"L {runway}")
    state_for = getattr(runtime, 'state_for', None)
    if state_for and res.get('ok'):
        state_for(cs).cleared = True


def _check_arm(sim, runtime, runway: str, armed: set,
                initial_star_len: dict, warmup_wpts: int = 2) -> None:
    """Arm planes that have consumed `warmup_wpts` STAR waypoints. Mirrors
    watch._check_arm with takeover_mode='after-wpts'.
    """
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
            _arm_plane(sim, runtime, cs, ac, runway)
            armed.add(cs)


def run_one_scenario(runtime,
                     seed: int,
                     spawn_rate: int,
                     max_steps: int,
                     airport: str = 'test',
                     runway: str = '27',
                     warmup_wpts: int = 2,
                     out_csv: Optional[Path] = None,
                     ) -> ScenarioResult:
    """Run a single multi-plane scenario with `runtime` driving all armed planes.

    Spawning is enabled at `spawn_rate` seconds. The sim runs until either:
      - crash_occurred (some pair collided)
      - max_steps elapsed

    Outcomes for each plane that was ever present:
      LANDED        — plane successfully landed
      IMPROPER_EXIT — plane left the radar area
      CRASHED       — plane was involved in the terminating collision
      TRUNCATED     — sim ended (crash or max_steps) before this plane terminated

    Per-plane violation_s counts integer seconds during which the plane had
    collision_warning=True under the strict 2-NM / 1000-ft rule.
    """
    _seed_all(seed)
    # Drop any state from prior scenarios on the same runtime (cleared
    # flags, density cache).
    if hasattr(runtime, 'reset'):
        runtime.reset()
    sim = SimulationEnv(radar_side=800, airport_name=airport,
                        spawn_rate=spawn_rate, star_mode=True)
    _install_strict_collision_rule(sim)

    if out_csv is not None:
        # in_memory=True buffers rows in a StringIO so we get ONE bulk
        # write at scenario end. Per-row volume flushes were dominating
        # wall time on Modal (~40x slowdown vs local).
        rec = HumanDataRecorder(spawn_single=False, in_memory=True)
        rec.start()
        sim.recorder = rec

    # Per-callsign in-flight bookkeeping (cleared on terminate).
    plane_spawn_t: dict[str, float] = {}
    plane_violation_s: dict[str, float] = {}
    completions: list[Traj] = []
    crashed_callsigns: set[str] = set()
    armed: set[str] = set()
    initial_star_len: dict[str, int] = {}

    crash_time: Optional[float] = None
    steps_done = 0

    for _ in range(max_steps):
        # --- Pre-step bookkeeping (counts the current second toward the
        # warning total for any plane currently flashing).
        for cs, ac in sim.aircraft_list.items():
            if cs not in plane_spawn_t:
                plane_spawn_t[cs] = sim.sim_time
                plane_violation_s[cs] = 0.0
            if ac.collision_warning:
                plane_violation_s[cs] += 1.0

        pre_keys = set(sim.aircraft_list.keys())
        pre_landed, pre_exit = sim.num_landed, sim.improper_exits

        # Arm planes that have consumed the warmup STAR waypoints, then
        # drive only the armed set (matches training-time semantics).
        _check_arm(sim, runtime, runway, armed, initial_star_len, warmup_wpts)
        runtime.tick(sim, armed=armed)
        sim.step(1.0)
        steps_done += 1

        if sim.crash_occurred:
            # Pick the plane(s) marked .crash and record them as CRASHED.
            for cs, ac in sim.aircraft_list.items():
                if ac.crash:
                    crashed_callsigns.add(cs)
                    completions.append(Traj(
                        callsign=cs, outcome='CRASHED',
                        spawn_t=plane_spawn_t.get(cs, sim.sim_time),
                        term_t=sim.sim_time,
                        violation_s=plane_violation_s.get(cs, 0.0)))
            crash_time = sim.sim_time
            break

        # Detect terminated planes (LANDED or IMPROPER_EXIT). Attribute
        # outcomes via the sim's counter deltas; single-removal-per-step
        # is overwhelmingly common, multi-removal falls back to UNKNOWN.
        post_keys = set(sim.aircraft_list.keys())
        removed = sorted(pre_keys - post_keys)
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

    # Any plane still in sim.aircraft_list when we exit the loop = TRUNCATED.
    # (Crashed planes were already added in the crash branch above; everything
    # else still flying is TRUNCATED.)
    for cs in sim.aircraft_list:
        if cs in crashed_callsigns:
            continue
        completions.append(Traj(
            callsign=cs, outcome='TRUNCATED',
            spawn_t=plane_spawn_t.get(cs, sim.sim_time),
            term_t=sim.sim_time,
            violation_s=plane_violation_s.get(cs, 0.0)))

    if sim.recorder is not None:
        # Bulk-write the buffered CSV to disk in one shot. Cheap.
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


def fair_cap(res_a: ScenarioResult, res_b: ScenarioResult,
              max_steps: int) -> tuple[list[Traj], list[Traj], float]:
    """Apply the fairness cap. Trajs from either side that terminated AFTER
    t_cap are re-classified TRUNCATED. Returns (kept_a, kept_b, t_cap).
    """
    inf = float(max_steps + 1)
    t_cap = min(res_a.crash_time or inf,
                res_b.crash_time or inf,
                float(max_steps))

    def apply(rs: list[Traj]):
        out = []
        for t in rs:
            if t.term_t <= t_cap:
                out.append(t)
            else:
                out.append(Traj(callsign=t.callsign, outcome='TRUNCATED',
                                spawn_t=t.spawn_t, term_t=min(t.term_t, t_cap),
                                violation_s=t.violation_s))
        return out

    return apply(res_a.trajs), apply(res_b.trajs), t_cap


def metrics_from(trajs: list[Traj]) -> dict:
    """Drop TRUNCATED, compute the three asked-for metrics over the rest."""
    kept = [t for t in trajs if t.outcome != 'TRUNCATED']
    n = len(kept)
    if n == 0:
        return {'n': 0, 'n_landed': 0, 'n_crashed': 0, 'n_exit': 0,
                'policy_sr': 0.0, 'crash_rate': 0.0,
                'avg_violation_s': 0.0}
    n_landed = sum(1 for t in kept if t.outcome == 'LANDED')
    n_crashed = sum(1 for t in kept if t.outcome == 'CRASHED')
    n_exit = sum(1 for t in kept if t.outcome == 'IMPROPER_EXIT')
    n_unknown = sum(1 for t in kept if t.outcome == 'UNKNOWN')
    non_crashed = n_landed + n_exit + n_unknown
    return {
        'n': n,
        'n_landed': n_landed,
        'n_crashed': n_crashed,
        'n_exit': n_exit,
        'n_unknown': n_unknown,
        'policy_sr': (n_landed / non_crashed) if non_crashed else 0.0,
        'crash_rate': n_crashed / n,
        'avg_violation_s': float(np.mean([t.violation_s for t in kept])),
    }


def compare(runtime_a, runtime_b,
            tag_a: str, tag_b: str,
            out_root: Path,
            n_target: int = 512,
            seed_base: int = 0,
            spawn_rate: int = 90,
            max_steps: int = 1500,
            max_scenarios: int = 200,
            verbose: bool = True,
            ) -> dict:
    """Run paired scenarios until each side has ≥ n_target kept trajs.

    out_root is a directory; CSVs go to out_root/<tag_a>/ and out_root/<tag_b>/.
    Returns a dict with full per-side stats + per-scenario records.
    """
    dir_a = out_root / tag_a
    dir_b = out_root / tag_b
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    all_kept_a: list[Traj] = []
    all_kept_b: list[Traj] = []
    scenarios: list[dict] = []

    n_scen = 0
    s = seed_base
    t_start = time.time()
    while ((len(all_kept_a) < n_target or len(all_kept_b) < n_target)
           and n_scen < max_scenarios):
        csv_a = dir_a / f'seed_{s:06d}.csv'
        csv_b = dir_b / f'seed_{s:06d}.csv'
        res_a = run_one_scenario(runtime_a, seed=s, spawn_rate=spawn_rate,
                                  max_steps=max_steps, out_csv=csv_a)
        res_b = run_one_scenario(runtime_b, seed=s, spawn_rate=spawn_rate,
                                  max_steps=max_steps, out_csv=csv_b)
        kept_a, kept_b, t_cap = fair_cap(res_a, res_b, max_steps)
        # Drop TRUNCATED here too so the accumulator is the keepable set.
        kept_a = [t for t in kept_a if t.outcome != 'TRUNCATED']
        kept_b = [t for t in kept_b if t.outcome != 'TRUNCATED']
        all_kept_a.extend(kept_a)
        all_kept_b.extend(kept_b)

        scenarios.append({
            'seed': s,
            'crash_time_a': res_a.crash_time,
            'crash_time_b': res_b.crash_time,
            't_cap': t_cap,
            'n_kept_a': len(kept_a),
            'n_kept_b': len(kept_b),
            'csv_a': str(csv_a.relative_to(out_root)),
            'csv_b': str(csv_b.relative_to(out_root)),
        })
        if verbose:
            elapsed = time.time() - t_start
            print(f'  [scenario {n_scen+1:3d}] seed={s} '
                  f'crash_a={res_a.crash_time} crash_b={res_b.crash_time} '
                  f't_cap={t_cap:.0f}  +A={len(kept_a):2d}  +B={len(kept_b):2d}  '
                  f'cum: A={len(all_kept_a)}/{n_target} B={len(all_kept_b)}/{n_target}  '
                  f'({elapsed:.0f}s)',
                  flush=True)
        s += 1
        n_scen += 1

    # Truncate to exactly n_target (in spawn-order) so both sides report on
    # the same population size.
    final_a = all_kept_a[:n_target]
    final_b = all_kept_b[:n_target]

    summary = {
        'config': {
            'tag_a': tag_a, 'tag_b': tag_b,
            'n_target': n_target, 'seed_base': seed_base,
            'spawn_rate': spawn_rate, 'max_steps': max_steps,
            'max_scenarios': max_scenarios,
            'n_scenarios_run': n_scen,
            'reached_target_a': len(all_kept_a) >= n_target,
            'reached_target_b': len(all_kept_b) >= n_target,
        },
        'metrics_a': metrics_from(final_a),
        'metrics_b': metrics_from(final_b),
        'scenarios': scenarios,
        'trajs_a': [t.to_dict() for t in final_a],
        'trajs_b': [t.to_dict() for t in final_b],
    }
    (out_root / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def build_runtime(ckpt: str, multi_ckpt: str | None = None,
                  bc_seed: str | None = None,
                  alt_floor_ft: float = 1000.0, runway: str = '27',
                  issue_speed: bool = True,
                  deterministic: bool = False):
    if multi_ckpt:
        return MultiRuntime(
            multi_ckpt_path=multi_ckpt,
            ppo_seed_ckpt_path=ckpt,
            bc_seed_path=bc_seed,
            alt_floor_ft=alt_floor_ft,
            runway=runway,
            issue_speed=issue_speed,
            deterministic_base=deterministic,
        )
    return Runtime(
        ckpt_path=ckpt,
        bc_seed_path=bc_seed,
        alt_floor_ft=alt_floor_ft,
        runway=runway,
        issue_speed=issue_speed,
        deterministic=deterministic,
    )


def _print_report(summary: dict) -> None:
    cfg = summary['config']
    mA, mB = summary['metrics_a'], summary['metrics_b']
    print()
    print('=' * 72)
    print(f' Paired comparison  ({cfg["tag_a"]}  vs  {cfg["tag_b"]})')
    print('=' * 72)
    print(f"  n_target          = {cfg['n_target']}")
    print(f"  scenarios run     = {cfg['n_scenarios_run']}  "
          f"(spawn_rate={cfg['spawn_rate']}s, max_steps={cfg['max_steps']}s)")
    print(f"  reached A={cfg['reached_target_a']} B={cfg['reached_target_b']}")
    print()
    cols = f"  {'metric':<22} {cfg['tag_a']:>16}    {cfg['tag_b']:>16}"
    print(cols)
    print('  ' + '-' * (len(cols) - 2))
    def row(name, va, vb, fmt='{:.2%}'):
        print(f"  {name:<22} {fmt.format(va):>16}    {fmt.format(vb):>16}")
    row('policy_sr  (L/non-C)', mA['policy_sr'], mB['policy_sr'])
    row('crash_rate (C/total)', mA['crash_rate'], mB['crash_rate'])
    row('avg violation_s', mA['avg_violation_s'], mB['avg_violation_s'],
        '{:.2f} s')
    row('n_landed',  mA['n_landed'],  mB['n_landed'],  '{:d}')
    row('n_crashed', mA['n_crashed'], mB['n_crashed'], '{:d}')
    row('n_exit',    mA['n_exit'],    mB['n_exit'],    '{:d}')
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ckpt-a', required=True,
                    help='Model A PPO ckpt (single GMM, or seed for multi).')
    ap.add_argument('--multi-ckpt-a', default=None,
                    help='If set, A is MultiRuntime with this radar head.')
    ap.add_argument('--ckpt-b', required=True)
    ap.add_argument('--multi-ckpt-b', default=None)
    ap.add_argument('--tag-a', default='model_a')
    ap.add_argument('--tag-b', default='model_b')
    ap.add_argument('--out', required=True,
                    help='Output dir (will create <tag>/seed_*.csv inside).')
    ap.add_argument('--n-target', type=int, default=512)
    ap.add_argument('--seed-base', type=int, default=0)
    ap.add_argument('--spawn-rate', type=int, default=90)
    ap.add_argument('--max-steps', type=int, default=1500)
    ap.add_argument('--max-scenarios', type=int, default=200)
    ap.add_argument('--bc-seed', default=None)
    ap.add_argument('--airport', default='test')
    ap.add_argument('--runway', default='27')
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()

    rt_a = build_runtime(args.ckpt_a, args.multi_ckpt_a, args.bc_seed,
                          deterministic=args.deterministic)
    rt_b = build_runtime(args.ckpt_b, args.multi_ckpt_b, args.bc_seed,
                          deterministic=args.deterministic)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = compare(
        rt_a, rt_b, args.tag_a, args.tag_b, out_root,
        n_target=args.n_target, seed_base=args.seed_base,
        spawn_rate=args.spawn_rate, max_steps=args.max_steps,
        max_scenarios=args.max_scenarios,
    )
    _print_report(summary)


if __name__ == '__main__':
    main()
