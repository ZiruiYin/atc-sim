"""Modal entrypoint — standalone eval for a multi-PPO ckpt.

Same termination conditions as PPO training (OUT_OF_ZONE on by default).
Writes the 6-pack to a fresh dir under the run's volume path; pulls
that dir back locally afterwards.

Usage:
    PYTHONIOENCODING=utf-8 modal run rl_multiple/modal_sanity_eval.py \\
        --ckpt-relpath runs_ppo_multi/phase1_v1/iter_0020.pt \\
        --out-suffix iter_0020_eval_50 \\
        --n-per-star 50
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rl_multiple.modal_config import (
    app, eval_multi_remote,
    REMOTE_VOLUME_MOUNT, VOLUME_NAME, _LOCAL_MULTI_RUNS, _MODAL_CMD,
)


@app.local_entrypoint()
def main(
    ckpt_relpath: str,
    out_suffix: str = '',
    ppo_seed_relpath: str = 'runs_ppo/continuous_03/iter_0160.pt',
    bc_seed_relpath: str = 'runs/bc_gmm_single_full/run_11/best.pt',
    n_per_star: int = 50,
    seed_base: int = 999_000,
    n_workers: int = 32,
    no_out_of_zone_terminate: bool = False,
    out_of_zone_max_consecutive: int = 10,
    everywhere_pen: float = 0.002,
    pull_back: bool = True,
):
    """ckpt_relpath: path on the Modal volume, e.g.
       runs_ppo_multi/phase1_v1/iter_0020.pt

       out_suffix:    name for the new sibling dir holding the 6-pack.
                       Default: <ckpt_stem>_eval_<n_per_star>"""
    ckpt_p = Path(ckpt_relpath)
    if not out_suffix:
        out_suffix = f'{ckpt_p.stem}_eval_{n_per_star}'
    out_relpath = str(ckpt_p.parent / out_suffix).replace('\\', '/')

    args = {
        'ckpt_relpath': ckpt_relpath,
        'out_dir_relpath': out_relpath,
        'ppo_seed_relpath': ppo_seed_relpath,
        'bc_seed_relpath': bc_seed_relpath,
        'n_per_star': n_per_star,
        'seed_base': seed_base,
        'n_workers': n_workers,
        'out_of_zone_terminate': not no_out_of_zone_terminate,
        'out_of_zone_max_consecutive': out_of_zone_max_consecutive,
        'everywhere_step_penalty': everywhere_pen,
    }
    print(f'  -- multi-PPO eval on Modal')
    print(f'  -- ckpt:    /{ckpt_relpath}')
    print(f'  -- out:     /{out_relpath}')
    print(f'  -- {n_per_star} cases per STAR x 6 STARs = {n_per_star * 6} total')

    result = eval_multi_remote.remote(args)

    o = result['overall']
    print()
    print('=' * 64)
    print(f' EVAL — {result["ckpt_relpath"]}')
    print('=' * 64)
    print(f' overall SR          = {o["success_rate"]*100:6.2f}%  '
          f'({o["n_success"]}/{o["n"]})')
    print(f' overall macro_green = {o["macro_pct_steps_in_green"]*100:6.2f}%')
    print()
    print(f' {"STAR":<8} {"SR%":>6}  {"green%":>7}  {"n":>4}')
    for star in ('NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3'):
        s = result['per_star'].get(star, {})
        if not s.get('n'):
            continue
        print(f' {star:<8} {s["success_rate"]*100:>5.1f}%  '
              f'{s["pct_steps_in_green"]*100:>6.1f}%  '
              f'{s["n"]:>4}')

    if pull_back:
        # Pull just the new 6-pack dir back locally for inspection
        # / PNG rendering.
        local_dir = _LOCAL_MULTI_RUNS / Path(out_relpath).relative_to('runs_ppo_multi')
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n  -- pulling /{out_relpath} → {local_dir}')
        subprocess.run(
            [*_MODAL_CMD, 'volume', 'get', '--force', VOLUME_NAME,
             f'/{out_relpath}', str(local_dir.parent)],
            check=True,
        )
        # Render PNGs locally
        try:
            from rl_multiple.eval_io import render_pngs
            n_s, n_f = render_pngs(local_dir)
            print(f'  [png] succ={n_s} fail={n_f}')
        except Exception as exc:
            print(f'  [png] render skipped: {type(exc).__name__}: {exc}')
