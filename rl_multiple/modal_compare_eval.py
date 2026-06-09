"""Modal entrypoint — paired multi-plane A/B eval.

Drives `rl_multiple.modal_config.compare_remote` (which does all the
remote work — defined in modal_config.py so Modal can resolve the
package import in the container without sys.path tricks).

Usage:
  PYTHONIOENCODING=utf-8 modal run rl_multiple/modal_compare_eval.py \\
      --out-relpath rollout_comparisons/baseline_vs_multi_02 \\
      --n-target 512 --spawn-rate 90 --max-steps 1500 --n-workers 32

Defaults compare the shipped single-GMM PPO (continuous_03 iter 160)
against the multi-PPO (continuous_03 + phase2/continuous_02 best.pt).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rl_multiple.modal_config import (
    app, compare_remote, VOLUME_NAME, _MODAL_CMD,
)


DEFAULT_PPO_SEED_RELPATH = 'runs_ppo/continuous_03/iter_0160.pt'
DEFAULT_BC_SEED_RELPATH = 'runs/bc_gmm_single_full/run_11/best.pt'
DEFAULT_MULTI_RELPATH = 'runs_ppo_multi/phase2/continuous_02/best.pt'


@app.local_entrypoint()
def main(
    out_relpath: str = 'rollout_comparisons/baseline_vs_multi_02',
    ckpt_a_relpath: str = DEFAULT_PPO_SEED_RELPATH,
    multi_ckpt_a_relpath: str = '',
    tag_a: str = 'baseline_singleppo',
    ckpt_b_relpath: str = DEFAULT_PPO_SEED_RELPATH,
    multi_ckpt_b_relpath: str = DEFAULT_MULTI_RELPATH,
    tag_b: str = 'multi_ppo',
    bc_seed_relpath: str = DEFAULT_BC_SEED_RELPATH,
    n_target: int = 512,
    seed_base: int = 0,
    spawn_rate: int = 90,
    max_steps: int = 1500,
    max_scenarios: int = 300,
    n_workers: int = 16,
    deterministic: bool = False,
    pull_back: bool = True,
):
    args = {
        'out_relpath': out_relpath,
        'ckpt_a_relpath': ckpt_a_relpath,
        'multi_ckpt_a_relpath': multi_ckpt_a_relpath or None,
        'tag_a': tag_a,
        'ckpt_b_relpath': ckpt_b_relpath,
        'multi_ckpt_b_relpath': multi_ckpt_b_relpath or None,
        'tag_b': tag_b,
        'bc_seed_relpath': bc_seed_relpath,
        'n_target': n_target,
        'seed_base': seed_base,
        'spawn_rate': spawn_rate,
        'max_steps': max_steps,
        'max_scenarios': max_scenarios,
        'n_workers': n_workers,
        'deterministic': deterministic,
    }

    print(f'  -- paired multi-plane comparison on Modal')
    print(f'  -- tag_a={tag_a}  ckpt_a={ckpt_a_relpath}'
          + (f'  +radar={multi_ckpt_a_relpath}' if multi_ckpt_a_relpath else ''))
    print(f'  -- tag_b={tag_b}  ckpt_b={ckpt_b_relpath}'
          + (f'  +radar={multi_ckpt_b_relpath}' if multi_ckpt_b_relpath else ''))
    print(f'  -- n_target={n_target}/side  '
          f'spawn_rate={spawn_rate}s  max_steps={max_steps}s  '
          f'max_scenarios={max_scenarios}')
    print(f'  -- workers={n_workers}')
    print(f'  -- writing to /{out_relpath} on Modal volume')

    result = compare_remote.remote(args)
    summary = result['summary']
    cfg = summary['config']
    mA, mB = summary['metrics_a'], summary['metrics_b']

    print()
    print('=' * 72)
    print(f' Paired comparison  ({cfg["tag_a"]}  vs  {cfg["tag_b"]})')
    print('=' * 72)
    print(f"  scenarios run     = {cfg['n_scenarios_run']}/{cfg['max_scenarios']}")
    print(f"  reached n_target  = "
          f"A:{cfg['reached_target_a']}  B:{cfg['reached_target_b']}")
    print()
    print(f"  {'metric':<24} {cfg['tag_a']:>20}    {cfg['tag_b']:>20}")
    print('  ' + '-' * 70)
    def row(name, va, vb, fmt):
        print(f"  {name:<24} {fmt.format(va):>20}    {fmt.format(vb):>20}")
    row('policy_sr   (L / non-C)', mA['policy_sr'], mB['policy_sr'], '{:.2%}')
    row('crash_rate  (C / total)', mA['crash_rate'], mB['crash_rate'], '{:.2%}')
    row('avg violation_s', mA['avg_violation_s'], mB['avg_violation_s'],
        '{:.2f} s')
    row('n_landed',  mA['n_landed'],  mB['n_landed'],  '{:d}')
    row('n_crashed', mA['n_crashed'], mB['n_crashed'], '{:d}')
    row('n_exit',    mA['n_exit'],    mB['n_exit'],    '{:d}')
    row('n_total',   mA['n'],         mB['n'],         '{:d}')
    print()

    if pull_back:
        local_dir = Path(out_relpath)
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f'  -- pulling /{out_relpath} -> {local_dir}')
        try:
            subprocess.run(
                [*_MODAL_CMD, 'volume', 'get', '--force', VOLUME_NAME,
                 f'/{out_relpath}', str(local_dir.parent)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f'  -- pull failed: {e}')
