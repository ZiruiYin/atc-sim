"""Modal entrypoint for the plan-mode eval.

Runs `heuristics_multiple.eval.run_one_scenario` across many scenarios
in parallel on Modal, collects kept trajectories until `n_target` is
reached, and writes per-scenario CSVs + summary.json + REPORT.md to a
volume directory. Local entrypoint pulls the directory back.

Usage:
  PYTHONIOENCODING=utf-8 modal run heuristics_multiple/modal_eval.py \\
      --out-relpath rollout_comparisons/plan_eval_01 \\
      --n-target 512 --max-scenarios 200 --n-workers 32

Defaults:
  ckpt          rl_ppo/runs/continuous_runs/continuous_03/best.pt (iter 160)
  spawn_rate    90 s
  max_steps     1500 s
  plan_steps    500 s         (also the per-plane lifetime cap)
  batch_size    3 trajs / iter
  iters         3 conflict-resolution rounds (max 9 trajs / plane)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "atc-heuristics-eval"
VOLUME_NAME = "atc-bc"
REMOTE_REPO_ROOT = "/root/atc-sim"
REMOTE_VOLUME_MOUNT = "/root/atc-sim/_modal"
_MODAL_CMD = [sys.executable, "-m", "modal"]


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO_ROOT,
        ignore=[
            "__pycache__", "*.pyc", ".git", ".vscode", ".pytest_cache",
            "rl_bc/cache", "rl_bc/runs", "rl_bc/_modal", "rl_bc/rollouts",
            "rl_ppo/runs", "rl_ppo/_modal",
            "rl_multiple/runs",
            "rl_multiple/rollout_comparisons",
            "heuristics_multiple/rollout_comparisons",
            "_internal", "*.exe", "ATC-Sim.exe",
            "human_data", "doc", "eval", "data_viz",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Worker chunk — runs a slice of scenarios sequentially in one mp.Pool process.
# Lives at module level so mp.Pool's spawn start_method can pickle it.
# --------------------------------------------------------------------------- #


def _eval_worker_chunk(args_tuple):
    """Run `seeds` sequentially; return (chunk_idx, kept_trajs, scenarios)."""
    import os
    import sys as _sys
    import time
    (chunk_idx, seeds, ckpt_path, bc_seed_path, out_root,
     spawn_rate, max_steps, plan_steps, max_steps_1_2, max_steps_3,
     batch_size, max_conflict_iters,
     airport, runway) = args_tuple

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REMOTE_REPO_ROOT)
    # Cap torch threads BEFORE first torch import so workers don't
    # oversubscribe BLAS across the Modal container's CPUs.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)

    from rl_multiple.runtime import Runtime
    from heuristics_multiple.eval import run_one_scenario
    from pathlib import Path as _Path

    out_root = _Path(out_root)
    csv_dir = out_root / 'scenarios'
    csv_dir.mkdir(parents=True, exist_ok=True)

    runtime = Runtime(
        ckpt_path=ckpt_path,
        bc_seed_path=bc_seed_path,
        alt_floor_ft=1000.0,
        runway=runway,
        issue_speed=True,
        deterministic=False,
    )

    kept_trajs = []
    scenarios = []
    for s in seeds:
        t0 = time.time()
        csv_path = csv_dir / f'seed_{s:06d}.csv'
        res = run_one_scenario(
            runtime, seed=s, spawn_rate=spawn_rate, max_steps=max_steps,
            plan_steps=plan_steps,
            max_steps_1_2=max_steps_1_2, max_steps_3=max_steps_3,
            batch_size=batch_size,
            max_conflict_iters=max_conflict_iters,
            airport=airport, runway=runway,
            out_csv=csv_path,
        )
        kept = [t.to_dict() for t in res.trajs
                if t.outcome not in ('TRUNCATED', 'UNKNOWN')]
        kept_trajs.extend(kept)
        scenarios.append({
            'seed': s,
            'crash_time': res.crash_time,
            'sim_time_end': res.sim_time_end,
            'n_kept': len(kept),
            'csv_path': str(csv_path.relative_to(out_root)),
        })
        print(f"  [chunk {chunk_idx} seed={s}] "
              f"crash={res.crash_time} +{len(kept)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return chunk_idx, kept_trajs, scenarios


# --------------------------------------------------------------------------- #
# Remote driver.
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    # 32 CPUs is right-sized for both paths:
    #   - parallel-scenarios: default n_workers=32 saturates the 32 CPUs
    #     (1 worker process per CPU).
    #   - single-sim: 1 main process + ~28 rollout pool processes ≈ 29
    #     active CPUs, ~90% utilization with a bit of headroom for the OS.
    # Allocating 64 CPUs wastes ~50% on single-sim and we can spend that
    # spend on additional parallel container launches instead.
    cpu=32,
    memory=32 * 1024,
    timeout=4 * 60 * 60,
    # Opt into non-preemptible workers (~3× cost) by setting
    # MODAL_NONPREEMPTIBLE=1 in the launch env. Only needed for runs
    # long enough that Modal's normal preemption loop derails them
    # (e.g. full_rollouts).
    nonpreemptible=os.environ.get('MODAL_NONPREEMPTIBLE') == '1',
)
def eval_plan_remote(args: dict):
    """Run plan-mode eval until `n_target` kept trajectories are collected.

    Two paths depending on `inner_pool_size`:
      * `inner_pool_size == 1` (default): parallel-scenarios. Spawns
        `n_workers` chunks via mp.Pool, each runs scenarios sequentially
        with serial replan inside.
      * `inner_pool_size > 1`: single-sim. Runs scenarios sequentially in
        the main process, parallelizes the per-replan rollouts via
        `heuristics_multiple.rollout.init_pool`.
    """
    import os
    import sys as _sys
    import multiprocessing as mp
    from pathlib import Path

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REMOTE_REPO_ROOT)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from heuristics_multiple.eval import compute_metrics, Traj

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    out_root = vol_root / args['out_relpath']
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(vol_root / args['ckpt_relpath'])
    bc_seed_path = (str(vol_root / args['bc_seed_relpath'])
                     if args.get('bc_seed_relpath') else None)

    n_target = int(args['n_target'])
    seed_base = int(args.get('seed_base', 0))
    max_scenarios = int(args.get('max_scenarios', 200))
    n_workers = int(args.get('n_workers', 32))
    # When set, the worker initializes a multiprocessing pool of this size
    # for the per-scenario rollouts inside `replan_all`. Implies
    # single-scenario mode (one sim at a time, parallel rollouts inside).
    inner_pool_size = int(args.get('inner_pool_size', 1))
    spawn_rate = int(args.get('spawn_rate', 90))
    max_steps = int(args.get('max_steps', 1500))
    plan_steps = int(args.get('plan_steps', 500))
    # Training-env-style pre-LOC TIMEOUT caps (rl_multiple/config.py:54-55).
    max_steps_1_2 = int(args.get('max_steps_1_2', 1200))
    max_steps_3 = int(args.get('max_steps_3', 500))
    batch_size = int(args.get('batch_size', 3))
    max_conflict_iters = int(args.get('max_conflict_iters', 3))
    # Full-rollouts mode: each candidate rollout runs until LANDED (not
    # truncated at plan_steps); INTERMEDIATE outcomes are rejected, so
    # every chosen plan is guaranteed to land. plan_steps becomes a hint
    # for conflict-detection horizon (we use full_rollout_max_steps as
    # the real bound).
    full_rollouts = bool(args.get('full_rollouts', False))
    full_rollout_max_steps = int(args.get('full_rollout_max_steps', 5000))
    airport = args.get('airport', 'test')
    runway = args.get('runway', '27')

    if inner_pool_size > 1:
        # Single-sim path: run scenarios sequentially in this container,
        # parallelize the per-replan rollouts via heuristics_multiple's
        # rollout pool.
        import time as _time
        from rl_multiple.runtime import Runtime
        from heuristics_multiple.eval import run_one_scenario
        from heuristics_multiple.rollout import init_pool, shutdown_pool

        # Cap torch threads BEFORE first torch import in any sub-process —
        # set env once, the inner pool inherits it.
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        import torch
        torch.set_num_threads(1)

        runtime = Runtime(
            ckpt_path=ckpt_path,
            bc_seed_path=bc_seed_path,
            alt_floor_ft=1000.0,
            runway=runway,
            issue_speed=True,
            deterministic=False,
        )

        print(f"[plan-eval] single-sim mode  inner_pool={inner_pool_size}  "
              f"n_target={n_target}  max_scenarios={max_scenarios}",
              flush=True)
        print(f"[plan-eval] warming inner rollout pool "
              f"({inner_pool_size} workers)...", flush=True)
        init_pool(runtime, n_workers=inner_pool_size, warm=True)
        print(f"[plan-eval] pool ready", flush=True)

        # Single-sim semantics: run ONE sim until it crashes OR until we
        # collect `n_target` kept trajectories (whichever first). On a
        # crash, restart with the next seed and keep going. Sims that
        # never crash will exit cleanly when stop_after_kept fires.
        seeds = list(range(seed_base, seed_base + max_scenarios))
        # CSVs go to a sibling `rollouts/...` directory whose nesting
        # mirrors the eval path. So out_relpath `heuristics_multiple/eval/
        # full_512/plan_steps_100` writes CSVs to `heuristics_multiple/
        # rollouts/full_512/plan_steps_100/`. Decouples raw trajectory
        # data from metadata (REPORT.md, summary.json) while keeping the
        # two trees structurally parallel.
        out_rel = args['out_relpath']
        prefix = 'heuristics_multiple/eval/'
        if out_rel.startswith(prefix):
            rollouts_rel = 'heuristics_multiple/rollouts/' + out_rel[len(prefix):]
        else:
            rollouts_rel = 'heuristics_multiple/rollouts/' + Path(out_rel).name
        csv_dir = vol_root / rollouts_rel
        csv_dir.mkdir(parents=True, exist_ok=True)
        all_kept = []
        all_scenarios = []
        try:
            for s in seeds:
                if len(all_kept) >= n_target:
                    print(f"[plan-eval] hit n_target={n_target} "
                          f"after {len(all_scenarios)} scenarios; stopping",
                          flush=True)
                    break
                remaining = n_target - len(all_kept)
                t0 = _time.time()
                csv_path = csv_dir / f'seed_{s:06d}.csv'
                res = run_one_scenario(
                    runtime, seed=s, spawn_rate=spawn_rate,
                    max_steps=max_steps,
                    plan_steps=plan_steps,
                    max_steps_1_2=max_steps_1_2, max_steps_3=max_steps_3,
                    batch_size=batch_size,
                    max_conflict_iters=max_conflict_iters,
                    airport=airport, runway=runway,
                    out_csv=csv_path,
                    stop_after_kept=remaining,
                    full_rollouts=full_rollouts,
                    full_rollout_max_steps=full_rollout_max_steps,
                )
                kept = [t.to_dict() for t in res.trajs
                        if t.outcome not in ('TRUNCATED', 'UNKNOWN')]
                all_kept.extend(kept)
                all_scenarios.append({
                    'seed': s,
                    'crash_time': res.crash_time,
                    'sim_time_end': res.sim_time_end,
                    'n_kept': len(kept),
                    # Path relative to repo root (mirrors the eval path).
                    'csv_path': f'{rollouts_rel}/seed_{s:06d}.csv',
                })
                why = 'crash' if res.crash_time else (
                    'enough kept' if len(all_kept) >= n_target else 'max_steps')
                print(f"  [seed={s}] {why}  t_end={res.sim_time_end:.0f}s  "
                      f"+{len(kept)}  cum={len(all_kept)}  "
                      f"({_time.time()-t0:.0f}s wall)", flush=True)
        finally:
            shutdown_pool()
    else:
        seeds = list(range(seed_base, seed_base + max_scenarios))
        chunks = [seeds[i::n_workers] for i in range(n_workers)]
        worker_args = [
            (i, ch, ckpt_path, bc_seed_path, str(out_root),
             spawn_rate, max_steps, plan_steps, max_steps_1_2, max_steps_3,
             batch_size, max_conflict_iters, airport, runway)
            for i, ch in enumerate(chunks) if ch
        ]
        print(f"[plan-eval] dispatching {len(worker_args)} workers, "
              f"{len(seeds)} seeds total, n_target={n_target}", flush=True)

        all_kept = []
        all_scenarios = []
        with mp.Pool(processes=len(worker_args)) as pool:
            for chunk_idx, kept, scn in pool.imap_unordered(
                    _eval_worker_chunk, worker_args):
                all_kept.extend(kept)
                all_scenarios.extend(scn)
                print(f"[plan-eval] chunk {chunk_idx} done: +{len(kept)}  "
                      f"cum={len(all_kept)}", flush=True)

    all_scenarios.sort(key=lambda r: r['seed'])
    final = all_kept[:n_target]
    metrics = compute_metrics([Traj(**t) for t in final])

    summary = {
        'config': {
            'ckpt': args['ckpt_relpath'],
            'n_target': n_target,
            'seed_base': seed_base,
            'max_scenarios': max_scenarios,
            'inner_pool_size': inner_pool_size,
            'mode': ('single_sim' if inner_pool_size > 1
                      else 'parallel_scenarios'),
            'spawn_rate': spawn_rate,
            'max_steps': max_steps,
            'plan_steps': plan_steps,
            'max_steps_1_2': max_steps_1_2,
            'max_steps_3': max_steps_3,
            'batch_size': batch_size,
            'max_conflict_iters': max_conflict_iters,
            'n_scenarios_run': len(all_scenarios),
            'reached_target': len(all_kept) >= n_target,
            'n_collected': len(all_kept),
        },
        'metrics': metrics,
        'scenarios': all_scenarios,
        'trajs': final,
    }

    summary_path = out_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2),
                             encoding='utf-8')

    # Auto-render a REPORT.md so the output is self-describing.
    report_md = _render_report(args, metrics, summary)
    (out_root / 'REPORT.md').write_text(report_md, encoding='utf-8')

    volume.commit()
    return {'summary': summary, 'out_relpath': args['out_relpath']}


def _render_report(args: dict, metrics: dict, summary: dict) -> str:
    cfg = summary['config']
    n = metrics['n']
    sr_pct = metrics['policy_sr'] * 100.0
    return f"""# Plan-mode eval — heuristics_multiple

**Driver:** `heuristics_multiple.modal_eval` → `eval_plan_remote` on Modal
**Code:** `heuristics_multiple/eval.py` (scenario runner),
`heuristics_multiple/flight_plan.py` (replan + conflict search),
`heuristics_multiple/rollout.py` (single-plane rollout)

## Method (one line)

For each scenario: spawn the policy-driven sim; whenever a plane crosses
warmup, request a `replan_all` that samples `batch_size={cfg['batch_size']}`
candidate trajectories per offender plane, runs a backtracking search
across the accumulated pool for a conflict-free assignment under the
2 NM / 1000 ft same-medium rule, and falls back to greedy max-separation
if `max_conflict_iters={cfg['max_conflict_iters']}` rounds don't find a
clean combination. Each plane then follows its plan deterministically
for up to `plan_steps={cfg['plan_steps']}` sim-seconds; planes that
exceed that lifetime without finishing are wiped as TIMEOUT.

## Config

| param | value |
|---|---|
| ckpt | `{cfg['ckpt']}` |
| spawn_rate | {cfg['spawn_rate']} s |
| max_steps | {cfg['max_steps']} s |
| plan_steps | {cfg['plan_steps']} s |
| max_steps_1_2 (pre-LOC cap, STAR1/2) | {cfg['max_steps_1_2']} s |
| max_steps_3 (pre-LOC cap, STAR3) | {cfg['max_steps_3']} s |
| batch_size | {cfg['batch_size']} |
| max_conflict_iters | {cfg['max_conflict_iters']} |
| n_target | {cfg['n_target']} |
| n_scenarios run | {cfg['n_scenarios_run']} |
| reached target | {cfg['reached_target']} |
| n_collected | {cfg['n_collected']} |

## Outcomes

```
LANDED        — touched down on runway 27
CRASHED       — terminating collision pair
IMPROPER_EXIT — flew out of radar bounds
TIMEOUT       — wiped by the plan_steps lifetime cap
TRUNCATED     — dropped before metrics (sim stopped first)
```

## Metric definitions

```
kept           = LANDED + CRASHED + IMPROPER_EXIT + TIMEOUT + LOC_ABOVE_GS
                 (TRUNCATED dropped)
policy_sr      = LANDED / kept
num_crashes    = absolute count
avg_violation_s = sum(violation_s over kept) / kept
```

## Results (n = {n} kept trajectories)

| metric | value |
|---|---|
| **policy_sr** | **{sr_pct:.2f} %** ({metrics['n_landed']} / {n}) |
| **num_crashes** | **{metrics['num_crashes']}** |
| **avg violation_s** | **{metrics['avg_violation_s']:.2f} s** |
| n_landed | {metrics['n_landed']} |
| n_crashed | {metrics['n_crashed']} |
| n_improper_exit | {metrics['n_improper_exit']} |
| n_timeout | {metrics['n_timeout']} |
| n_loc_above_gs | {metrics['n_loc_above_gs']} |

## Files

```
{cfg.get('out_relpath', args['out_relpath'])}/
├── REPORT.md      ← this file
├── summary.json   ← config, metrics, per-scenario records, all kept trajs
└── scenarios/
    └── seed_NNNNNN.csv  (× {cfg['n_scenarios_run']})  HumanDataRecorder dump
```
"""


# --------------------------------------------------------------------------- #
# Local entrypoint.
# --------------------------------------------------------------------------- #


DEFAULT_CKPT_RELPATH = 'runs_ppo/continuous_03/iter_0160.pt'


@app.local_entrypoint()
def main(
    out_relpath: str = 'rollout_comparisons/plan_eval_01',
    ckpt_relpath: str = DEFAULT_CKPT_RELPATH,
    bc_seed_relpath: str = 'runs/bc_gmm_single_full/run_11/best.pt',
    n_target: int = 512,
    seed_base: int = 0,
    # Effectively unlimited — the actual stopping condition is hitting
    # `n_target` kept trajectories. The only reason to ever cap this is
    # to bound the worst-case Modal runtime if something goes wrong.
    max_scenarios: int = 10000,
    n_workers: int = 32,
    inner_pool_size: int = 1,
    spawn_rate: int = 90,
    max_steps: int = 1500,
    plan_steps: int = 500,
    max_steps_1_2: int = 1200,
    max_steps_3: int = 500,
    batch_size: int = 3,
    max_conflict_iters: int = 3,
    full_rollouts: bool = False,
    full_rollout_max_steps: int = 5000,
    pull_back: bool = True,
):
    args = {
        'out_relpath': out_relpath,
        'ckpt_relpath': ckpt_relpath,
        'bc_seed_relpath': bc_seed_relpath,
        'n_target': n_target,
        'seed_base': seed_base,
        'max_scenarios': max_scenarios,
        'n_workers': n_workers,
        'inner_pool_size': inner_pool_size,
        'spawn_rate': spawn_rate,
        'max_steps': max_steps,
        'plan_steps': plan_steps,
        'max_steps_1_2': max_steps_1_2,
        'max_steps_3': max_steps_3,
        'full_rollouts': full_rollouts,
        'full_rollout_max_steps': full_rollout_max_steps,
        'batch_size': batch_size,
        'max_conflict_iters': max_conflict_iters,
    }

    print(f'  -- plan-mode eval on Modal')
    print(f'  -- ckpt = {ckpt_relpath}')
    print(f'  -- n_target = {n_target}  max_scenarios = {max_scenarios}  '
          f'workers = {n_workers}')
    print(f'  -- spawn_rate = {spawn_rate}s  max_steps = {max_steps}s  '
          f'plan_steps = {plan_steps}s  '
          f'pre_loc_cap_1_2 = {max_steps_1_2}s  pre_loc_cap_3 = {max_steps_3}s  '
          f'batch = {batch_size}  iters = {max_conflict_iters}')
    print(f'  -- writing to /{out_relpath} on Modal volume')

    result = eval_plan_remote.remote(args)
    metrics = result['summary']['metrics']
    cfg = result['summary']['config']

    print()
    print('=' * 72)
    print(f"  plan-mode eval — {cfg['n_collected']} kept trajectories"
          f" (target {cfg['n_target']}, reached={cfg['reached_target']})")
    print('=' * 72)
    print(f"  policy_sr        = {metrics['policy_sr']*100:.2f}%  "
          f"({metrics['n_landed']} landed / {metrics['n']} kept)")
    print(f"  num_crashes      = {metrics['num_crashes']}")
    print(f"  avg violation_s  = {metrics['avg_violation_s']:.2f} s")
    print(f"  n_landed         = {metrics['n_landed']}")
    print(f"  n_crashed        = {metrics['n_crashed']}")
    print(f"  n_improper_exit  = {metrics['n_improper_exit']}")
    print(f"  n_timeout        = {metrics['n_timeout']}")
    print()

    if pull_back:
        # Pull both the eval folder (REPORT.md + summary.json) and the
        # rollouts folder (per-scenario CSVs) — paths mirror each other
        # under heuristics_multiple/{eval,rollouts}/.
        prefix = 'heuristics_multiple/eval/'
        if out_relpath.startswith(prefix):
            rollouts_rel = 'heuristics_multiple/rollouts/' + out_relpath[len(prefix):]
        else:
            rollouts_rel = 'heuristics_multiple/rollouts/' + Path(out_relpath).name
        for rel in (out_relpath, rollouts_rel):
            local_dir = Path(rel)
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f'  -- pulling /{rel} -> {local_dir}')
            try:
                subprocess.run(
                    [*_MODAL_CMD, 'volume', 'get', '--force', VOLUME_NAME,
                     f'/{rel}', str(local_dir.parent)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f'  -- pull failed: {e}')
