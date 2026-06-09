"""Modal entrypoint for the fair model-driven eval.

Mirrors `heuristics_multiple/modal_eval.py` structure exactly but drives
planes with a `Runtime` / `MultiRuntime` (not flight plans). Single sim
per Modal app; restart only on natural crash; TIMEOUT enforced via the
pre-LOC step cap; same CSV format and metric definitions as `full_512_v2`.

Two model tags are supported via the `--tag` arg:
  baseline_singleppo  — Runtime(continuous_03 PPO only, no radar head)
  multi_ppo           — MultiRuntime(continuous_03 + phase-2 radar head)

Usage (one Modal app per model, both run in parallel):
  MODAL_NONPREEMPTIBLE=1 PYTHONIOENCODING=utf-8 modal run --detach \\
      rl_multiple/modal_eval_fair.py \\
      --out-relpath rl_multiple/eval/baseline_vs_multi_v2/baseline_singleppo \\
      --tag baseline_singleppo \\
      --n-target 512 --max-steps 1000000

  MODAL_NONPREEMPTIBLE=1 PYTHONIOENCODING=utf-8 modal run --detach \\
      rl_multiple/modal_eval_fair.py \\
      --out-relpath rl_multiple/eval/baseline_vs_multi_v2/multi_ppo \\
      --tag multi_ppo \\
      --n-target 512 --max-steps 1000000
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "atc-rl-multi-eval-fair"
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
            "rl_multiple/rollouts",
            "rl_multiple/eval",
            "heuristics_multiple/rollout_comparisons",
            "heuristics_multiple/eval",
            "heuristics_multiple/rollouts",
            "_internal", "*.exe", "ATC-Sim.exe",
            "human_data", "doc", "eval", "data_viz",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Ckpts (paths on the Modal volume, set by training pipelines):
DEFAULT_PPO_CKPT_RELPATH = 'runs_ppo/continuous_03/iter_0160.pt'
DEFAULT_MULTI_CKPT_RELPATH = 'runs_ppo_multi/phase2/continuous_02/best.pt'
DEFAULT_BC_SEED_RELPATH = 'runs/bc_gmm_single_full/run_11/best.pt'


# --------------------------------------------------------------------------- #
# Remote driver.
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    cpu=16,
    memory=32 * 1024,
    timeout=6 * 60 * 60,
    # Opt into non-preemptible workers (~3× cost) by setting
    # MODAL_NONPREEMPTIBLE=1 in the launch env.
    nonpreemptible=os.environ.get('MODAL_NONPREEMPTIBLE') == '1',
)
def eval_fair_remote(args: dict):
    """Single-sim model-driven eval until n_target kept trajectories."""
    import os as _os
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

    _os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REMOTE_REPO_ROOT)

    _os.environ.setdefault('OMP_NUM_THREADS', '1')
    _os.environ.setdefault('MKL_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)

    from rl_multiple.eval_fair import run_one_scenario, compute_metrics, Traj
    from rl_multiple.compare_eval import build_runtime

    vol_root = _Path(REMOTE_VOLUME_MOUNT)
    out_root = vol_root / args['out_relpath']
    out_root.mkdir(parents=True, exist_ok=True)

    tag = args['tag']
    ckpt_path = str(vol_root / args['ckpt_relpath'])
    multi_ckpt_path = (str(vol_root / args['multi_ckpt_relpath'])
                       if args.get('multi_ckpt_relpath') else None)
    bc_seed_path = (str(vol_root / args['bc_seed_relpath'])
                     if args.get('bc_seed_relpath') else None)

    n_target = int(args['n_target'])
    seed_base = int(args.get('seed_base', 0))
    max_scenarios = int(args.get('max_scenarios', 10000))
    spawn_rate = int(args.get('spawn_rate', 90))
    max_steps = int(args.get('max_steps', 1_000_000))
    max_steps_1_2 = int(args.get('max_steps_1_2', 1200))
    max_steps_3 = int(args.get('max_steps_3', 500))
    airport = args.get('airport', 'test')
    runway = args.get('runway', '27')
    warmup_wpts = int(args.get('warmup_wpts', 2))

    print(f"  -- fair eval on Modal  tag={tag}")
    print(f"  -- ckpt = {args['ckpt_relpath']}")
    if multi_ckpt_path:
        print(f"  -- multi_ckpt = {args['multi_ckpt_relpath']}")
    print(f"  -- n_target = {n_target}  max_scenarios = {max_scenarios}")
    print(f"  -- spawn_rate = {spawn_rate}s  max_steps = {max_steps}s  "
          f"pre_loc_cap_1_2 = {max_steps_1_2}s  pre_loc_cap_3 = {max_steps_3}s",
          flush=True)
    print(f"  -- writing to /{args['out_relpath']} on Modal volume", flush=True)

    runtime = build_runtime(
        ckpt=ckpt_path,
        multi_ckpt=multi_ckpt_path,
        bc_seed=bc_seed_path,
        alt_floor_ft=1000.0,
        runway=runway,
        issue_speed=True,
        deterministic=False,
    )

    # Two layout conventions:
    #   - `rl_multiple/eval/<rest>` → split: CSVs go to `rl_multiple/rollouts/<rest>/`
    #   - anything else (e.g. `rollout_comparisons/<run>/<tag>`) → co-located:
    #     CSVs live in the same out_root folder as summary.json + REPORT.md.
    out_rel = args['out_relpath']
    prefix = 'rl_multiple/eval/'
    if out_rel.startswith(prefix):
        rollouts_rel = 'rl_multiple/rollouts/' + out_rel[len(prefix):]
    else:
        rollouts_rel = out_rel  # CSVs sibling to summary.json
    csv_dir = vol_root / rollouts_rel
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fair-eval] single-sim mode  n_target={n_target}  "
          f"max_scenarios={max_scenarios}", flush=True)

    seeds = list(range(seed_base, seed_base + max_scenarios))
    all_kept = []
    all_scenarios = []

    for s in seeds:
        if len(all_kept) >= n_target:
            print(f"[fair-eval] hit n_target={n_target} "
                  f"after {len(all_scenarios)} scenarios; stopping",
                  flush=True)
            break
        remaining = n_target - len(all_kept)
        t0 = _time.time()
        csv_path = csv_dir / f'seed_{s:06d}.csv'
        res = run_one_scenario(
            runtime, seed=s, spawn_rate=spawn_rate,
            max_steps=max_steps,
            max_steps_1_2=max_steps_1_2, max_steps_3=max_steps_3,
            airport=airport, runway=runway,
            warmup_wpts=warmup_wpts,
            out_csv=csv_path,
            stop_after_kept=remaining,
        )
        kept = [t.to_dict() for t in res.trajs
                if t.outcome not in ('TRUNCATED', 'UNKNOWN')]
        all_kept.extend(kept)
        all_scenarios.append({
            'seed': s,
            'crash_time': res.crash_time,
            'sim_time_end': res.sim_time_end,
            'n_kept': len(kept),
            'csv_path': f'{rollouts_rel}/seed_{s:06d}.csv',
        })
        why = ('crash' if res.crash_time
               else ('enough kept' if len(all_kept) >= n_target
                     else 'max_steps'))
        print(f"  [seed={s}] {why}  t_end={res.sim_time_end:.0f}s  "
              f"+{len(kept)}  cum={len(all_kept)}  "
              f"({_time.time()-t0:.0f}s wall)", flush=True)

    all_scenarios.sort(key=lambda r: r['seed'])
    final = all_kept[:n_target]
    metrics = compute_metrics([Traj(**t) for t in final])

    summary = {
        'config': {
            'tag': tag,
            'ckpt': args['ckpt_relpath'],
            'multi_ckpt': args.get('multi_ckpt_relpath'),
            'n_target': n_target,
            'seed_base': seed_base,
            'max_scenarios': max_scenarios,
            'mode': 'single_sim',
            'spawn_rate': spawn_rate,
            'max_steps': max_steps,
            'max_steps_1_2': max_steps_1_2,
            'max_steps_3': max_steps_3,
            'warmup_wpts': warmup_wpts,
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

    report_md = _render_report(args, metrics, summary)
    (out_root / 'REPORT.md').write_text(report_md, encoding='utf-8')

    volume.commit()
    return {'summary': summary, 'out_relpath': args['out_relpath']}


def _render_report(args: dict, metrics: dict, summary: dict) -> str:
    cfg = summary['config']
    n = metrics['n']
    sr_pct = metrics['policy_sr'] * 100.0
    multi_line = (f"| multi_ckpt | `{cfg['multi_ckpt']}` |\n"
                   if cfg.get('multi_ckpt') else '')
    return f"""# Fair model-driven eval — `{cfg['tag']}`

**Driver:** `rl_multiple.modal_eval_fair` → `eval_fair_remote` on Modal
**Code:** `rl_multiple/eval_fair.py` (scenario runner; reuses
`heuristics_multiple.eval` helpers for arming, TIMEOUT enforcement, and
the strict collision rule)

## Method

Single `SimulationEnv` per scenario. Arming happens after `warmup_wpts`
STAR waypoints (training-time semantics). Then the runtime drives every
armed plane each tick via `runtime.tick(sim, armed=armed)`. The pre-LOC
step cap is enforced — a plane that has been armed for more than
`max_steps_1_2={cfg['max_steps_1_2']}` ticks (STAR1/2) or
`max_steps_3={cfg['max_steps_3']}` ticks (STAR3) without intercepting LOC
is force-removed and recorded as TIMEOUT. A scenario ends on natural sim
crash or when `n_target={cfg['n_target']}` kept trajectories accumulate;
`max_steps={cfg['max_steps']}` is a safety belt (never fires normally).

When a sim crashes, a fresh `SimulationEnv` is started with the next
seed — the only way a new sim begins. `n_sims = 1` means a single
uninterrupted sim hit `n_target` without crashing.

## Config

| param | value |
|---|---|
| tag | `{cfg['tag']}` |
| ckpt | `{cfg['ckpt']}` |
{multi_line}| spawn_rate | {cfg['spawn_rate']} s |
| max_steps | {cfg['max_steps']} s (safety belt) |
| max_steps_1_2 (pre-LOC cap, STAR1/2) | {cfg['max_steps_1_2']} s |
| max_steps_3 (pre-LOC cap, STAR3) | {cfg['max_steps_3']} s |
| warmup_wpts | {cfg['warmup_wpts']} |
| n_target | {cfg['n_target']} |
| n_scenarios run | {cfg['n_scenarios_run']} |
| reached target | {cfg['reached_target']} |
| n_collected | {cfg['n_collected']} |

## Outcomes

```
LANDED        — touched down on runway 27
CRASHED       — terminating collision pair
IMPROPER_EXIT — flew out of radar bounds
TIMEOUT       — wiped by the pre-LOC step cap (plane couldn't intercept LOC)
TRUNCATED     — dropped before metrics (sim ended before this plane finished)
```

## Metric definitions

```
kept           = LANDED + CRASHED + IMPROPER_EXIT + TIMEOUT
policy_sr      = LANDED / kept
num_crashes    = absolute count of CRASHED in kept
avg_violation_s = sum(violation_s) / kept  (live-runner authoritative)
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

## Files

```
{args['out_relpath']}/
├── REPORT.md      ← this file
└── summary.json   ← config, metrics, per-scenario records, all kept trajs

rl_multiple/rollouts/{args['out_relpath'][len('rl_multiple/eval/'):] if args['out_relpath'].startswith('rl_multiple/eval/') else args['out_relpath']}/
└── seed_NNNNNN.csv (× {cfg['n_scenarios_run']})
```
"""


# --------------------------------------------------------------------------- #
# Local entrypoint.
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def main(
    out_relpath: str = 'rl_multiple/eval/baseline_vs_multi_v2/baseline_singleppo',
    tag: str = 'baseline_singleppo',
    ckpt_relpath: str = DEFAULT_PPO_CKPT_RELPATH,
    multi_ckpt_relpath: str = '',     # set to non-empty for multi_ppo
    bc_seed_relpath: str = DEFAULT_BC_SEED_RELPATH,
    n_target: int = 512,
    seed_base: int = 0,
    max_scenarios: int = 10000,
    spawn_rate: int = 90,
    max_steps: int = 1_000_000,
    max_steps_1_2: int = 1200,
    max_steps_3: int = 500,
    warmup_wpts: int = 2,
    pull_back: bool = True,
):
    # Sanity: if tag implies the radar head, require multi_ckpt; if not,
    # forbid it. Bail early so we don't accidentally label one as the
    # other.
    if tag == 'multi_ppo' and not multi_ckpt_relpath:
        raise SystemExit(
            "tag=multi_ppo requires --multi-ckpt-relpath; refusing to launch "
            "a baseline run under that label.")
    if tag == 'baseline_singleppo' and multi_ckpt_relpath:
        raise SystemExit(
            "tag=baseline_singleppo conflicts with --multi-ckpt-relpath; "
            "refusing to launch a multi run under that label.")

    args = {
        'out_relpath': out_relpath,
        'tag': tag,
        'ckpt_relpath': ckpt_relpath,
        'multi_ckpt_relpath': multi_ckpt_relpath or None,
        'bc_seed_relpath': bc_seed_relpath,
        'n_target': n_target,
        'seed_base': seed_base,
        'max_scenarios': max_scenarios,
        'spawn_rate': spawn_rate,
        'max_steps': max_steps,
        'max_steps_1_2': max_steps_1_2,
        'max_steps_3': max_steps_3,
        'warmup_wpts': warmup_wpts,
    }

    print(f'  -- fair model-driven eval on Modal')
    print(f'  -- tag = {tag}')
    print(f'  -- ckpt = {ckpt_relpath}')
    if multi_ckpt_relpath:
        print(f'  -- multi_ckpt = {multi_ckpt_relpath}')
    print(f'  -- n_target = {n_target}  max_scenarios = {max_scenarios}')
    print(f'  -- spawn_rate = {spawn_rate}s  max_steps = {max_steps}s  '
          f'pre_loc_cap_1_2 = {max_steps_1_2}s  pre_loc_cap_3 = {max_steps_3}s')
    print(f'  -- writing to /{out_relpath} on Modal volume')

    result = eval_fair_remote.remote(args)
    metrics = result['summary']['metrics']
    cfg = result['summary']['config']

    print()
    print('=' * 72)
    print(f"  fair eval [{tag}] — {cfg['n_collected']} kept trajectories"
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
        prefix = 'rl_multiple/eval/'
        if out_relpath.startswith(prefix):
            rollouts_rel = 'rl_multiple/rollouts/' + out_relpath[len(prefix):]
            pull_paths = (out_relpath, rollouts_rel)
        else:
            # Co-located layout: CSVs live in out_relpath alongside
            # summary.json + REPORT.md — one pull does it all.
            pull_paths = (out_relpath,)
        for rel in pull_paths:
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
