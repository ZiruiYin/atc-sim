"""Local entrypoint for ONE multi-plane PPO BLOCK on Modal.

Mirrors `rl_ppo/modal_continuous.py`. Each invocation runs `[block_from,
block_to)` iters on Modal, saves to the volume, and pulls the run dir
back to `rl_multiple/runs/<run_name>/` afterwards.

Workflow (manually driven):

    # Block 1 — Phase 1 stabilization, iters 0..10 from frozen GMM seed
    PYTHONIOENCODING=utf-8 modal run rl_multiple/modal_continuous.py \\
        --run-name phase1_v1 \\
        --block-from-iter 0 --block-to-iter 10 \\
        --gamma 0.999 --lr-actor 1e-5 --lr-critic 3e-5 \\
        --entropy-coef 0.01 --target-kl 0.02

    # Inspect rl_multiple/runs/phase1_v1/log.jsonl for per-iter SR
    # before picking next-block knobs.

    # Block 2 — resume from iter_0010.pt, train to iter 30
    PYTHONIOENCODING=utf-8 modal run rl_multiple/modal_continuous.py \\
        --run-name phase1_v1 \\
        --block-from-iter 10 --block-to-iter 30 \\
        --gamma 0.999 --lr-actor 1e-5 --lr-critic 3e-5
"""
from __future__ import annotations

from rl_multiple.modal_config import (
    app, pull_multi_run_back, train_block_and_eval_multi,
)


# Defaults pin to the shipped single-plane PPO ckpt on the volume.
DEFAULT_PPO_CKPT_RELPATH = 'runs_ppo/continuous_03/iter_0160.pt'
DEFAULT_BC_SEED_RELPATH = 'runs/bc_gmm_single_full/run_11/best.pt'


@app.local_entrypoint()
def main(
    run_name: str,
    block_from_iter: int = 0,
    block_to_iter: int = 10,
    ppo_ckpt_relpath: str = DEFAULT_PPO_CKPT_RELPATH,
    bc_seed_relpath: str = DEFAULT_BC_SEED_RELPATH,
    # PPO hyperparams
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    lr_actor: float = 1e-5,
    lr_critic: float = 3e-5,
    clip_epsilon: float = 0.2,
    target_kl: float = 0.02,
    entropy_coef: float = 0.01,
    batch_size: int = 256,
    n_epochs: int = 4,
    n_rollouts_per_iter: int = 128,
    n_workers: int = 0,
    # Reward shaping (Phase 1 defaults match the ship)
    everywhere_pen: float = 0.001,
    step_pen_cap: float = 0.005,
    step_pen_per_nm: float = 0.0005,
    out_of_zone_terminate: bool = False,
    out_of_zone_max_consecutive: int = 5,
    per_star_sr_scale: float = 0.0,
    loop_penalty: float = 0.0,
    # Delta head clamps
    delta_hdg_clamp_deg: float = 30.0,
    delta_alt_clamp_kft: float = 1.0,
    delta_spd_clamp_kt: float = 30.0,
    delta_log_sigma_init: float = -1.5,
    delta_hidden: int = 64,
    # Misc
    seed: int = 0,
    no_wandb: bool = False,
    eval_six_pack_every: int = 20,
    save_every: int = 10,
    # Phase-2 multi-plane
    multi_plane: bool = False,
    spawn_rate: int = 120,
    collision_penalty: float = 0.10,
    init_radar_head_from: str = '',
):
    if not run_name:
        raise ValueError('--run-name is required')

    args = {
        'run_name': run_name,
        'block_from_iter': block_from_iter,
        'block_to_iter': block_to_iter,
        'ppo_ckpt_relpath': ppo_ckpt_relpath,
        'bc_seed_relpath': bc_seed_relpath,
        # PPO
        'gamma': gamma, 'gae_lambda': gae_lambda,
        'lr_actor': lr_actor, 'lr_critic': lr_critic,
        'clip_epsilon': clip_epsilon, 'target_kl': target_kl,
        'entropy_coef': entropy_coef, 'batch_size': batch_size,
        'n_epochs': n_epochs,
        'n_rollouts_per_iter': n_rollouts_per_iter,
        'n_workers': n_workers,
        # Reward
        'everywhere_step_penalty': everywhere_pen,
        'step_penalty_cap': step_pen_cap,
        'step_penalty_per_nm': step_pen_per_nm,
        'out_of_zone_terminate': out_of_zone_terminate,
        'out_of_zone_max_consecutive': out_of_zone_max_consecutive,
        'per_star_sr_scale': per_star_sr_scale,
        'loop_penalty_per_step': loop_penalty,
        # Delta head
        'delta_hdg_clamp_deg': delta_hdg_clamp_deg,
        'delta_alt_clamp_kft': delta_alt_clamp_kft,
        'delta_spd_clamp_kt': delta_spd_clamp_kt,
        'delta_log_sigma_init': delta_log_sigma_init,
        'delta_hidden': delta_hidden,
        # Misc
        'seed': seed,
        'use_wandb': not no_wandb,
        'wandb_group': 'ppo_multi',
        'eval_six_pack_every': eval_six_pack_every,
        'save_every': save_every,
        # Phase 2
        'multi_plane': multi_plane,
        'spawn_rate': spawn_rate,
        'collision_warning_penalty': collision_penalty,
        'init_radar_head_from': init_radar_head_from,
    }

    print(f"  -- multi-PPO run: {run_name}  block {block_from_iter}->{block_to_iter}")
    print(f"  -- frozen GMM seed: /{ppo_ckpt_relpath}")
    print(f"  -- bc seed (arch):  /{bc_seed_relpath}")
    print(f"  -- gamma={gamma}  lr_actor={lr_actor}  target_kl={target_kl}")
    print(f"  -- rollouts/iter={n_rollouts_per_iter}  entropy_coef={entropy_coef}")
    if block_from_iter == 0:
        print(f"  -- starting from frozen GMM (delta head zero-init)")
    else:
        print(f"  -- resuming from /runs_ppo_multi/{run_name}/"
              f"iter_{block_from_iter:04d}.pt")

    result = train_block_and_eval_multi.remote(args)
    print(f"\nremote summary: {result['train_summary']}")

    print(f"\n  -- pulling /runs_ppo_multi/{run_name} back to "
          f"rl_multiple/runs/{run_name}/ ...")
    local_dir = pull_multi_run_back(run_name)
    print(f"  [ok] ready: {local_dir}")

    # Render successful/unsuccessful PNGs locally — skipped for
    # multi-plane runs (user spec: replay CSVs only, no PNGs).
    if not multi_plane:
        try:
            from rl_multiple.eval_io import render_pngs
            n_rendered = 0
            for eval_dir in sorted(local_dir.glob('iter_*_eval')):
                succ_png = eval_dir / 'successful_trajectories.png'
                fail_png = eval_dir / 'unsuccessful_trajectories.png'
                if succ_png.exists() and fail_png.exists():
                    continue
                try:
                    n_s, n_f = render_pngs(eval_dir)
                    n_rendered += 1
                    print(f"  [png] {eval_dir.name}  succ={n_s} fail={n_f}")
                except Exception as exc:
                    print(f"  [png] {eval_dir.name} FAILED: "
                          f"{type(exc).__name__}: {exc}")
            if n_rendered:
                print(f"  [ok] rendered PNGs for {n_rendered} eval dirs")
        except ImportError as exc:
            print(f"  [warn] PNG render skipped (matplotlib?): {exc}")

    # Multi-plane: keep only the longest replay CSV per session set as
    # a representative sample, delete the rest to save disk.
    if multi_plane:
        replay_dir = local_dir / 'replays'
        if replay_dir.exists():
            csvs = sorted(replay_dir.glob('*.csv'),
                          key=lambda p: p.stat().st_size, reverse=True)
            if csvs:
                longest = csvs[0]
                print(f"  [replay] longest CSV: {longest.name}  "
                      f"({longest.stat().st_size/1024:.0f} KB; "
                      f"keeping; deleting {len(csvs)-1} others)")
                for c in csvs[1:]:
                    try:
                        c.unlink()
                    except OSError:
                        pass
