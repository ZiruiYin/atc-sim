"""bc_eval — Modal rollout + scoring for BC checkpoints, unified.

One command does the whole eval for any BC checkpoint that lives on the
`atc-bc` Modal volume (under `runs/<config>/run_N/best.pt`):

  1. ROLLOUT (Modal, CPU-heavy): 200 rollouts per STAR × 6 STARs = 1200
     episodes, using the SAME per-STAR step caps + LOC_BELOW_GS success
     rule as `rl_bc.eval.runner` / the PPO eval. The remote container is
     sized like the PPO rollout pool — `cpu=64` (~32 physical cores on
     EPYC) with a 32-worker process pool (cpu_count // 2), so each worker
     lands on its own physical core (see rl_ppo/modal_config.py).
  2. SAVE: pulls results back and writes the standard artifact set into
     `bc_eval/results/<config>/`:
         rollouts.csv          one row per episode
         trajectories.npz      per-step (a, c) — keys: stars, seeds,
                               outcomes, lengths, a_concat, c_concat
         summary.json          per-STAR + overall outcome aggregates
         raw.json              per-episode dicts (no position arrays)
  3. SCORE (local): runs `rl_ppo.eval_metrics.compute_eval_metrics` over
     that dir and writes `eval_metrics.json` — the eval CATEGORIES:
         success_rate (per-STAR + overall), pct_steps_in_green (green-zone
         coverage), pct_within_length_range, length_mean/std/var, and the
         outcome split (LOC_ABOVE_GS / LOC_BEHIND_THR / TIMEOUT / ...).

The rollouts.csv + trajectories.npz it writes are exactly the pair
`rl_ppo.eval_metrics` consumes, so you can also re-score any results dir
standalone with:  `python -m rl_ppo.eval_metrics bc_eval/results/<config>`

Usage (atc-sim env):
    modal run bc_eval/run_bc_eval.py --config bc_gmm_single_full
    modal run bc_eval/run_bc_eval.py --config bc_gmm_single_full_nodistill
    modal run bc_eval/run_bc_eval.py --config bc_gmm_single_full_human --run 1
    modal run bc_eval/run_bc_eval.py --config bc_fm_single --cases 200

Score-only (no Modal, re-score an existing results dir):
    python bc_eval/run_bc_eval.py --score-only bc_eval/results/bc_gmm_single_full
"""

from __future__ import annotations

import csv as csv_module
import json
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable regardless of launch style (`modal run
# bc_eval/...` vs a bare `python bc_eval/run_bc_eval.py --score-only`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_bc.modal_config import (
    ROLLOUT_CPU,
    ROLLOUT_WORKERS,
    app,
    rollout_remote,
)


CSV_FIELDS = ('star', 'seed', 'outcome', 'callsign', 'sim_time',
              'steps_used', 'error_type', 'error')


def _save_trajectories_npz(npz_path: Path, results: list[dict]) -> None:
    """Per-step (a, c) cache, schema identical to
    rl_bc.eval.viz_trajectories / rl_ppo.eval_runner so eval_metrics reads
    it. Inlined (numpy-only) to avoid pulling matplotlib."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    star_list = [r['star'] for r in results]
    seed_list = [int(r['seed']) for r in results]
    outcome_list = [r.get('outcome', '?') for r in results]
    lengths = np.array([len(r.get('a_traj', [])) for r in results],
                       dtype=np.int64)
    if results:
        a_concat = np.concatenate([np.asarray(r.get('a_traj', []), np.float32)
                                   for r in results]) if lengths.sum() else np.empty(0, np.float32)
        c_concat = np.concatenate([np.asarray(r.get('c_traj', []), np.float32)
                                   for r in results]) if lengths.sum() else np.empty(0, np.float32)
    else:
        a_concat = np.empty(0, np.float32)
        c_concat = np.empty(0, np.float32)
    np.savez_compressed(
        npz_path,
        stars=np.array(star_list),
        seeds=np.array(seed_list, dtype=np.int64),
        outcomes=np.array(outcome_list),
        lengths=lengths,
        a_concat=a_concat,
        c_concat=c_concat,
    )


def _write_artifacts(out_dir: Path, summary: dict, results: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'rollouts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv_module.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    raw_for_json = [{k: v for k, v in r.items() if k not in ('a_traj', 'c_traj')}
                    for r in results]
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out_dir / 'raw.json').write_text(json.dumps(raw_for_json, indent=2))
    _save_trajectories_npz(out_dir / 'trajectories.npz', results)


def _score_and_print(out_dir: Path) -> dict:
    """Run the canonical scorer and echo the eval categories."""
    from rl_ppo.eval_metrics import compute_eval_metrics
    m = compute_eval_metrics(out_dir, save=True)
    o = m['overall']
    print(f"\n=== eval categories ({out_dir}) ===")
    print(f"overall: SR={o['success_rate']*100:.1f}%  "
          f"green={o['macro_pct_steps_in_green']*100:.1f}%  "
          f"in_range={o['macro_pct_within_length_range']*100:.1f}%  "
          f"(n={o['n']})")
    print(f"{'STAR':<8}{'SR%':>7}{'green%':>8}{'in_rng%':>9}"
          f"{'len_mean':>10}{'len_std':>9}")
    for star, s in m['per_star'].items():
        if s['n'] == 0:
            continue
        print(f"{star:<8}{s['success_rate']*100:>6.1f}"
              f"{s['pct_steps_in_green']*100:>8.1f}"
              f"{s['pct_within_length_range']*100:>9.1f}"
              f"{s['length_mean']:>10.0f}{s['length_std']:>9.0f}")
    print(f"\n  eval_metrics.json → {out_dir / 'eval_metrics.json'}")
    return m


@app.local_entrypoint()
def main(
    config: str,                 # required, e.g. bc_gmm_single_full_nodistill
    run: int = 0,                # 0 → latest run on volume
    cases: int = 200,            # rollouts per STAR (×6 = total)
    max_steps: int = 0,          # 0 ⇒ per-STAR caps (1200/1200/500 ...)
    warmup_wpts: int = 2,
    runway: str = '27',
    airport: str = 'test',
    seed_base: int = 10_000,     # offset from training-rollout seed space
    out_dir: str = '',           # default bc_eval/results/<config>
):
    out = Path(out_dir) if out_dir else Path('bc_eval/results') / config
    print(f"  → dispatching rollout to Modal "
          f"(cpu={ROLLOUT_CPU}, workers={ROLLOUT_WORKERS}, config={config}, "
          f"run={'latest' if run <= 0 else run}, "
          f"{cases}×6={cases*6} rollouts)")
    res = rollout_remote.remote({
        "config_name": config,
        "run_n": (None if run <= 0 else int(run)),
        "cases": int(cases),
        "max_steps": int(max_steps),
        "warmup_wpts": int(warmup_wpts),
        "runway": str(runway),
        "airport": str(airport),
        "workers": ROLLOUT_WORKERS,
        "seed_base": int(seed_base),
    })
    summary, results = res["summary"], res["results"]
    print(f"\nremote ckpt: {res['ckpt_path']}")
    print(f"wall time  : {summary['_meta']['wall_seconds']:.1f}s on "
          f"{summary['_meta']['workers']} workers")
    _write_artifacts(out, summary, results)
    _score_and_print(out)


def _cli_score_only() -> bool:
    """Allow `python bc_eval/run_bc_eval.py --score-only <dir>` without Modal."""
    if '--score-only' in sys.argv:
        i = sys.argv.index('--score-only')
        target = sys.argv[i + 1] if i + 1 < len(sys.argv) else ''
        if not target:
            raise SystemExit("--score-only needs a results dir path")
        _score_and_print(Path(target))
        return True
    return False


if __name__ == '__main__':
    # `modal run` imports this module and calls main(); a bare
    # `python run_bc_eval.py --score-only <dir>` re-scores locally.
    if not _cli_score_only():
        print(__doc__)
