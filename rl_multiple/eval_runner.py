"""Standalone evaluation for a multi-plane PPO checkpoint.

Loads a multi-PPO ckpt (stores `delta_head_state`, optionally
`critic_state`), runs N fresh episodes per STAR through the same
MultiPPOEnv + CombinedPolicy stack used in training (so termination
logic is identical), and writes a 6-pack to the chosen out_dir:

    rollouts.csv
    trajectories.npz
    eval_metrics.json   ← SR, macro_green per STAR + overall
    summary.json
    raw.json
    successful_trajectories.png   (if matplotlib available)
    unsuccessful_trajectories.png

Termination conditions are inherited from rl_ppo.env.PPOEnv. To pin
them to a particular config (e.g. OUT_OF_ZONE on with 10-step
tolerance), pass via CLI or via the args dict.

CLI:
    python -m rl_multiple.eval_runner \\
        --ckpt rl_multiple/runs/phase1_v1/iter_0020.pt \\
        --out-dir rl_multiple/runs/phase1_v1/iter_0020_eval_50 \\
        --n-per-star 50
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import torch

from rl_multiple.config import MultiPPOConfig
from rl_multiple.eval_io import render_pngs, write_eval_six_pack
from rl_multiple.policy import CombinedPolicy
from rl_multiple.rollout import collect_rollouts, make_rollout_pool
from rl_multiple.runtime import resolve_bc_seed_path


DEFAULT_BC_SEED = Path('rl_bc/runs/bc_gmm_single_full/run_11/best.pt')


def run_eval(ckpt_path: str | Path,
             ppo_seed_ckpt: str | Path,
             out_dir: str | Path,
             *,
             n_per_star: int = 50,
             bc_seed: str | Path | None = None,
             seed_base: int = 999_000,
             n_workers: int = 0,
             out_of_zone_terminate: bool = True,
             out_of_zone_max_consecutive: int = 10,
             everywhere_step_penalty: float = 0.002,
             step_penalty_cap: float = 0.0,
             step_penalty_per_nm: float = 0.0,
             loop_penalty_per_step: float = 0.0,
             render: bool = True) -> dict:
    """Run N rollouts per STAR and write the 6-pack.

    `ckpt_path` is the multi-PPO ckpt (delta_head_state + critic_state).
    `ppo_seed_ckpt` is the FROZEN GMM the training started from
        (e.g. rl_ppo/runs/continuous_runs/continuous_03/best.pt).
    Returns the eval_metrics dict (with `overall` + `per_star`).
    """
    ckpt_path = Path(ckpt_path)
    ppo_seed_ckpt = Path(ppo_seed_ckpt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f'multi-PPO ckpt not found: {ckpt_path}')
    if not ppo_seed_ckpt.exists():
        raise FileNotFoundError(f'frozen-GMM seed not found: {ppo_seed_ckpt}')

    # Resolve BC seed (arch + standardizer) from the frozen-GMM ckpt if
    # not explicitly given.
    if bc_seed is None:
        blob = torch.load(ppo_seed_ckpt, map_location='cpu',
                          weights_only=False)
        bc_seed_path = resolve_bc_seed_path(blob)
    else:
        bc_seed_path = Path(bc_seed)
    if not bc_seed_path.exists():
        raise FileNotFoundError(f'BC seed not found: {bc_seed_path}')

    stars = ('NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3')

    # Worker config — mirrors what train.py passes. Reward overrides
    # are forwarded so OUT_OF_ZONE_TERMINATE etc. take effect inside
    # the spawn workers (those re-import reward_zones fresh).
    cfg_dict = {
        'airport_name': 'test',
        'runway': '27',
        'warmup_wpts': 2,
        'max_timesteps_star_1_2': 1200,
        'max_timesteps_star_3': 500,
        'success_reward': 10.0,
        'failure_reward': -10.0,
        'gs_capture_buffer_ft': 50.0,
        'density_cutoff_nm': 10.0,
        'density_n_bins': 36,
        'delta_hidden': 64,
        'value_hidden': 64,
        'value_dropout': 0.0,
        # Loop detector (defaults match c03 except disabled here)
        'loop_penalty_per_step': loop_penalty_per_step,
        'loop_prox_radius_nm': 0.75,
        'loop_min_gap_steps': 45,
        'loop_min_detour_nm': 1.0,
        # Reward overrides
        'reward_everywhere_step_penalty': everywhere_step_penalty,
        'reward_step_penalty_cap': step_penalty_cap,
        'reward_step_penalty_per_nm': step_penalty_per_nm,
        'reward_early_zone_multiplier': 1.0,
        'reward_early_window_steps': 1000,
        'reward_early_drift_penalty': 0.0,
        'reward_clean_terminal_threshold': 0.0,
        'reward_drifty_success_value': 0.0,
        'reward_out_of_zone_terminate': out_of_zone_terminate,
        'reward_out_of_zone_max_consecutive': out_of_zone_max_consecutive,
        'reward_per_star_sr_scale': 0.0,
        'reward_per_star_recent_sr': None,
    }

    if n_workers <= 0:
        import os
        n_workers = max(1, (os.cpu_count() or 2) // 2)

    print(f"[eval] ckpt           = {ckpt_path}")
    print(f"[eval] frozen GMM     = {ppo_seed_ckpt}")
    print(f"[eval] BC seed (arch) = {bc_seed_path}")
    print(f"[eval] {n_per_star} cases x {len(stars)} STARs = "
          f"{n_per_star * len(stars)} total  workers={n_workers}")

    pool = make_rollout_pool(ppo_seed_ckpt, bc_seed_path, cfg_dict,
                              n_workers)

    # Build a master CombinedPolicy with the frozen GMM, then overlay
    # the trained delta head + critic from the multi-PPO ckpt.
    policy = CombinedPolicy.from_ppo_ckpt(
        ppo_seed_ckpt, bc_seed_path=bc_seed_path,
        density_n_bins=cfg_dict['density_n_bins'],
        delta_hidden=cfg_dict['delta_hidden'],
        value_hidden=cfg_dict['value_hidden'],
        device='cpu',
    )
    multi_blob = torch.load(ckpt_path, map_location='cpu',
                             weights_only=False)
    if 'delta_head_state' not in multi_blob:
        raise ValueError(
            f"{ckpt_path}: not a multi-PPO ckpt "
            f"(no delta_head_state; keys={list(multi_blob.keys())[:6]})"
        )
    policy.delta_head.load_state_dict(multi_blob['delta_head_state'])
    if 'critic_state' in multi_blob:
        policy.critic.load_state_dict(multi_blob['critic_state'])

    try:
        trajs = collect_rollouts(
            pool, n_rollouts=n_per_star * len(stars),
            stars=stars, seed_offset=seed_base,
            policy_state=policy.state_dict_split(),
            verbose=True,
        )
    finally:
        pool.close()
        pool.join()

    metrics = write_eval_six_pack(trajs, out_dir, score=True)

    if render:
        try:
            n_s, n_f = render_pngs(out_dir)
            print(f"[eval] PNGs rendered: succ={n_s}  fail={n_f}")
        except Exception as exc:
            print(f"[eval] PNG render skipped: "
                  f"{type(exc).__name__}: {exc}")

    o = metrics['overall']
    print()
    print(f"  SR overall          = {o['success_rate']*100:.2f}%  "
          f"({o['n_success']}/{o['n']})")
    print(f"  macro_green overall = "
          f"{o['macro_pct_steps_in_green']*100:.2f}%")
    print()
    print(f"  {'STAR':<8} {'SR%':>6}  {'green%':>7}  {'n':>4}")
    for star in stars:
        s = metrics['per_star'].get(star, {})
        if not s.get('n'):
            continue
        print(f"  {star:<8} {s['success_rate']*100:>5.1f}%  "
              f"{s['pct_steps_in_green']*100:>6.1f}%  "
              f"{s['n']:>4}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ckpt', type=str, required=True,
                    help='Path to a multi-PPO ckpt (iter_NNNN.pt or '
                         'best.pt with delta_head_state).')
    ap.add_argument('--out-dir', type=str, required=True,
                    help='Where to write the 6-pack.')
    ap.add_argument('--ppo-seed-ckpt', type=str,
                    default=str(MultiPPOConfig.ppo_ckpt.default
                                if hasattr(MultiPPOConfig.ppo_ckpt, 'default')
                                else 'rl_ppo/runs/continuous_runs/continuous_03/best.pt'),
                    help='Frozen-GMM seed the multi-PPO ckpt was trained on.')
    ap.add_argument('--bc-seed', type=str, default=None,
                    help='BC seed (arch + standardizer). Default: resolved '
                         "from the PPO seed ckpt's config.actor_ckpt.")
    ap.add_argument('--n-per-star', type=int, default=50)
    ap.add_argument('--seed-base', type=int, default=999_000)
    ap.add_argument('--n-workers', type=int, default=0)
    ap.add_argument('--no-out-of-zone-terminate', action='store_true',
                    help='Disable OUT_OF_ZONE termination for eval. '
                         'Default: on (matches Phase 1 training).')
    ap.add_argument('--out-of-zone-max-consecutive', type=int, default=10)
    ap.add_argument('--everywhere-pen', type=float, default=0.002)
    ap.add_argument('--no-render', action='store_true',
                    help='Skip PNG rendering (use when matplotlib is '
                         'unavailable, e.g. inside a Modal container).')
    args = ap.parse_args()

    seed_default = MultiPPOConfig().ppo_ckpt
    ppo_seed = args.ppo_seed_ckpt if args.ppo_seed_ckpt else str(seed_default)

    run_eval(
        ckpt_path=args.ckpt,
        ppo_seed_ckpt=ppo_seed,
        out_dir=args.out_dir,
        n_per_star=args.n_per_star,
        bc_seed=args.bc_seed,
        seed_base=args.seed_base,
        n_workers=args.n_workers,
        out_of_zone_terminate=not args.no_out_of_zone_terminate,
        out_of_zone_max_consecutive=args.out_of_zone_max_consecutive,
        everywhere_step_penalty=args.everywhere_pen,
        render=not args.no_render,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
