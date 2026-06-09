"""End-of-block eval for a PPO checkpoint.

What this does:
  1. Take a PPO checkpoint (`iter_NNNN.pt` saved by `rl_ppo.train`)
     and wrap its `actor_state` into a BC-format checkpoint in a temp
     file (BC eval expects keys `model_state`, `standardizer_mean/std`,
     `config`). The standardizer + config come from the BC SEED actor
     the PPO run started from — recoverable from the PPO ckpt's
     `config.actor_ckpt` field.
  2. Call `rl_bc.eval.runner.run_eval_parallel` against that wrapped
     ckpt for `cases` rollouts × 6 STARs.
  3. Write `rollouts.csv`, `trajectories.npz`, `summary.json` into the
     output dir (same schema as existing eval folders).
  4. Score with `rl_ppo.eval_metrics.compute_eval_metrics`, writing
     `eval_metrics.json`.

Designed to be called from `rl_ppo/modal_continuous.py` after each
training block lands its checkpoint on the Modal volume. Runs entirely
on whatever CPUs the caller provides — no matplotlib (no PNGs).

CLI:
    python -m rl_ppo.eval_runner <ppo_ckpt.pt> <out_dir> [--cases 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _wrap_ppo_to_bc(ppo_ckpt: Path, dst_path: Path) -> Path:
    """Create a BC-format checkpoint at `dst_path` from a PPO ckpt.

    The PPO ckpt has `actor_state` (the rl_bc actor weights) and a
    config dict carrying the path to the BC seed it started from.
    The BC eval runner needs `model_state` + `standardizer_*` +
    `config` (with input_indices/radar_side/nm_range). We fish those
    out of the BC seed and assemble the wrapped file.
    """
    blob = torch.load(ppo_ckpt, map_location='cpu', weights_only=False)
    if 'actor_state' not in blob:
        raise ValueError(
            f"ckpt at {ppo_ckpt} is not a PPO checkpoint "
            f"(no `actor_state` key; got {list(blob.keys())[:6]}...)"
        )
    # Locate the BC seed used to start the PPO run. The PPO ckpt's
    # `config` field carries the path as a string. On Modal the path
    # is absolute under /root/atc-sim/_modal — local-scored ckpts get
    # the local path. We try both.
    seed_path_str = blob.get('config', {}).get('actor_ckpt', '')
    if not seed_path_str:
        raise RuntimeError(
            f"PPO ckpt has no `config.actor_ckpt` — can't recover BC seed"
        )
    seed_path = _resolve_seed_path(seed_path_str)
    seed_blob = torch.load(seed_path, map_location='cpu',
                           weights_only=False)
    wrapped = {
        'epoch': int(blob.get('iter', 0)),
        'model_state': blob['actor_state'],
        'standardizer_mean': seed_blob['standardizer_mean'],
        'standardizer_std': seed_blob['standardizer_std'],
        'config': seed_blob['config'],
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapped, dst_path)
    return dst_path


def _save_trajectories_npz(cache_path: Path, results: list[dict]) -> None:
    """Per-step (a, c) cache. Same key schema as
    `rl_bc.eval.viz_trajectories.save_trajectories` so the existing
    `rl_ppo.eval_metrics` and BC viz tools can read this file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    star_list = [r['star'] for r in results]
    seed_list = [int(r['seed']) for r in results]
    outcome_list = [r.get('outcome', '?') for r in results]
    lengths = np.array([len(r.get('a_traj', [])) for r in results],
                       dtype=np.int64)
    if n:
        a_concat = np.concatenate([np.asarray(r.get('a_traj', []),
                                              dtype=np.float32)
                                   for r in results])
        c_concat = np.concatenate([np.asarray(r.get('c_traj', []),
                                              dtype=np.float32)
                                   for r in results])
    else:
        a_concat = np.empty(0, np.float32)
        c_concat = np.empty(0, np.float32)
    np.savez_compressed(
        cache_path,
        stars=np.array(star_list),
        seeds=np.array(seed_list, dtype=np.int64),
        outcomes=np.array(outcome_list),
        lengths=lengths,
        a_concat=a_concat,
        c_concat=c_concat,
    )


def _resolve_seed_path(s: str) -> Path:
    """Translate a stored BC-seed path (possibly Modal-absolute) into
    something on the current filesystem."""
    p = Path(s)
    if p.exists():
        return p
    # Strip a common Modal prefix and try the repo-relative form.
    s_norm = s.replace('\\', '/')
    markers = [
        '/_modal/runs/',     # Modal volume layout: /…/_modal/runs/...
        '/runs/',
    ]
    for m in markers:
        if m in s_norm:
            tail = 'rl_bc/runs/' + s_norm.split(m, 1)[1]
            cand = Path(tail)
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"could not resolve BC seed ckpt referenced by PPO config: {s}"
    )


def run_eval(ppo_ckpt: str | Path,
             out_dir: str | Path,
             cases: int = 200,
             warmup_wpts: int = 2,
             runway: str = '27',
             airport: str = 'test',
             workers: int = 0,
             seed_base: int = 10_000,
             max_steps: int = 0) -> dict:
    """Evaluate `ppo_ckpt` and write results into `out_dir`.

    Returns the dict produced by `eval_metrics.compute_eval_metrics`.
    """
    from rl_bc.eval.runner import run_eval_parallel
    from rl_ppo.eval_metrics import compute_eval_metrics

    ppo_ckpt = Path(ppo_ckpt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if workers <= 0:
        import os
        workers = max(1, (os.cpu_count() or 2) // 2)

    # Wrap the PPO ckpt to BC format in a tmp file. Keep the tmp file
    # inside out_dir so failures are debuggable; remove after eval.
    wrapped_path = out_dir / '_wrapped_for_eval.pt'
    _wrap_ppo_to_bc(ppo_ckpt, wrapped_path)

    csv_path = out_dir / 'rollouts.csv'
    summary_path = out_dir / 'summary.json'
    raw_path = out_dir / 'raw.json'
    traj_path = out_dir / 'trajectories.npz'

    summary, results = run_eval_parallel(
        ckpt_path=wrapped_path,
        cases=int(cases),
        max_steps=int(max_steps),
        warmup_wpts=int(warmup_wpts),
        runway=runway,
        airport=airport,
        workers=int(workers),
        seed_base=int(seed_base),
        config_name='ppo_eval',
        csv_path=csv_path,
    )

    # raw.json (no position arrays) + summary.json
    raw_for_json = [{k: v for k, v in r.items()
                     if k not in ('a_traj', 'c_traj')}
                    for r in results]
    summary_path.write_text(json.dumps(summary, indent=2))
    raw_path.write_text(json.dumps(raw_for_json, indent=2))

    # trajectories.npz — schema matches rl_bc.eval.viz_trajectories.
    # We inline the save here instead of importing it because the
    # viz module pulls matplotlib at the top, which isn't in the
    # Modal PPO image. The save logic itself is just numpy.
    _save_trajectories_npz(traj_path, results)

    # Score it.
    metrics = compute_eval_metrics(out_dir, save=True)

    # Clean up the wrapped ckpt to avoid confusing future readers
    # (it's regenerable from the source PPO ckpt anyway).
    try:
        wrapped_path.unlink()
    except OSError:
        pass

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ppo_ckpt', type=str,
                    help='Path to a PPO checkpoint (iter_NNNN.pt).')
    ap.add_argument('out_dir', type=str,
                    help='Where to write rollouts.csv / trajectories.npz / '
                         'eval_metrics.json.')
    ap.add_argument('--cases', type=int, default=200,
                    help='Rollouts per STAR. Default 200 (= 1200 total).')
    ap.add_argument('--workers', type=int, default=0,
                    help='0 = autodetect cpu_count // 2 (skip SMT).')
    ap.add_argument('--seed-base', type=int, default=10_000,
                    help='Seed-base for the eval rollouts. Default '
                         '10000 so eval seeds do not collide with the '
                         'training-rollout seed space.')
    args = ap.parse_args()

    m = run_eval(args.ppo_ckpt, args.out_dir,
                 cases=args.cases, workers=args.workers,
                 seed_base=args.seed_base)
    # Compact stdout summary so the Modal log shows the key numbers
    # without having to fetch the JSON.
    o = m['overall']
    print(f"\n[eval_runner] {args.ppo_ckpt}")
    print(f"  overall: SR={o['success_rate']*100:.1f}%  "
          f"macro_green={o['macro_pct_steps_in_green']*100:.1f}%  "
          f"macro_in_range={o['macro_pct_within_length_range']*100:.1f}%")
    for star, s in m['per_star'].items():
        if s['n'] == 0:
            continue
        print(f"  {star}: SR={s['success_rate']*100:.1f}%  "
              f"green={s['pct_steps_in_green']*100:.1f}%  "
              f"in_range={s['pct_within_length_range']*100:.1f}%  "
              f"len_mean={s['length_mean']:.0f}  "
              f"len_std={s['length_std']:.0f}")


if __name__ == '__main__':
    main()
