"""Modal entrypoint — sanity-eval a PPO ckpt by running N rollouts/STAR
through PPOEnv (same termination logic as PPO training).

No disk writes. Reports per-STAR SR + outcome counts only.

Usage:
    PYTHONIOENCODING=utf-8 modal run rl_ppo/modal_sanity_eval.py \\
        --ckpt-relpath runs_ppo/continuous_03/iter_0160.pt \\
        --n-per-star 50
"""
from __future__ import annotations

from rl_ppo.modal_config import app, eval_sanity_remote


@app.local_entrypoint()
def main(
    ckpt_relpath: str = 'runs_ppo/continuous_03/iter_0160.pt',
    bc_seed_relpath: str = 'runs/bc_gmm_single_full/run_11/best.pt',
    n_per_star: int = 50,
    seed_base: int = 999_000,
    n_workers: int = 32,
):
    args = {
        'ckpt_relpath': ckpt_relpath,
        'bc_seed_relpath': bc_seed_relpath,
        'n_per_star': n_per_star,
        'seed_base': seed_base,
        'n_workers': n_workers,
    }
    print(f'  -- launching sanity eval on Modal '
          f'(no disk writes, no trajectory overwrite)')
    print(f'  -- ckpt: {ckpt_relpath}')
    print(f'  -- {n_per_star} cases per STAR x 6 STARs = '
          f'{n_per_star * 6} total')

    result = eval_sanity_remote.remote(args)

    print()
    print('=' * 60)
    print(f' SANITY EVAL — {result["ckpt_relpath"]}  iter={result["ckpt_iter"]}')
    print('=' * 60)
    print(f' overall SR: {result["sr_overall"] * 100:6.2f}%  '
          f'({result["n_success"]}/{result["total"]})')
    print()
    print(f' {"STAR":<8} {"succ":>5} {"n":>5} {"SR%":>7}   outcomes')
    for star in ('NORTH1', 'NORTH2', 'NORTH3',
                 'SOUTH1', 'SOUTH2', 'SOUTH3'):
        s = result['per_star'].get(star, {'n': 0, 'succ': 0, 'outcomes': {}})
        sr = s['succ'] / max(1, s['n']) * 100
        outcomes = ', '.join(f'{k}={v}' for k, v in
                             sorted(s['outcomes'].items(),
                                    key=lambda kv: -kv[1]))
        print(f' {star:<8} {s["succ"]:>5} {s["n"]:>5} {sr:>6.1f}%   '
              f'[{outcomes}]')

    crashes = result.get('crash_details', [])
    if crashes:
        print()
        print(f' {len(crashes)} CRASHED episode(s):')
        for c in crashes:
            print(f'   {c["star"]:<8} step {c["steps"]:>4}  '
                  f'-> {c["error"]}')
