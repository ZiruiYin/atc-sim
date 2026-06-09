"""Local entrypoint for Modal-hosted PPO training.

Allocates a fresh `run_N` slot under `/runs_ppo/` on the Modal volume,
dispatches the training, pulls the result back into `rl_ppo/runs/run_N/`.

The actor checkpoint is referenced by its volume-relative path under
`/runs/...` (the BC pipeline already syncs `rl_bc/runs/` to the volume —
the latest checkpoint should already be there). If you re-train BC, run
`modal run rl_bc/modal_train.py` first so the new checkpoint lands on
the volume before kicking PPO.

Usage:
    modal run rl_ppo/modal_train.py
    modal run rl_ppo/modal_train.py --n-iters 500 --n-rollouts 192
    modal run rl_ppo/modal_train.py --actor-run 10 --bc-config bc_gmm_single
    modal run rl_ppo/modal_train.py --no-wandb
"""
from __future__ import annotations

from pathlib import Path

from rl_ppo.modal_config import (
    app,
    next_ppo_run_number,
    pull_ppo_run_back,
    train_ppo_remote,
)


@app.local_entrypoint()
def main(
    # Actor seed selection — defaults to bc_gmm_single_full/run_11 (the
    # 7-D soft-DAgger model that's the current BC baseline).
    bc_config: str = 'bc_gmm_single_full',
    actor_run: int = 11,
    # PPO knobs you'll commonly tune.
    n_iters: int = 0,                   # 0 → use PPOConfig default
    n_rollouts: int = 0,                # 0 → use PPOConfig default
    n_workers: int = 0,                 # 0 → autodetect cpu_count // 2
    n_epochs: int = 0,
    lr_actor: float = 0.0,
    lr_critic: float = 0.0,
    clip_eps: float = 0.0,
    gamma: float = 0.0,
    gae_lambda: float = 0.0,
    target_kl: float = 0.0,
    entropy_coef: float = -1.0,         # -1 → use default (0.0)
    batch_size: int = 0,
    max_steps_1_2: int = 0,
    max_steps_3: int = 0,
    seed: int = 0,
    # In-run periodic eval — the cheap alternative to the block-at-a-time
    # continuous driver when you HOLD hyperparams constant (which is what
    # we ended up doing). `--eval-every K` runs N iters in ONE modal
    # invocation and evals every K iters into iter_NNNN_eval/. Equivalent
    # to the block method for the optimization (optimizer state + LR are
    # continuous either way); the ONLY difference is per_star_sr_scale's
    # recent_SR is frozen here (no mid-run refresh) — set it to 0 for a
    # clean ablation, or use modal_continuous.py if you need it adaptive.
    eval_every: int = 0,
    eval_cases: int = 200,
    # Reward-shaping knobs (mirror modal_continuous.py) so a straight run
    # can reproduce a recipe. Sentinels: floats < 0 / ints <= 0 = leave
    # PPOConfig default. Note 0.0 IS a valid value (forwarded).
    everywhere_pen: float = -1.0,
    step_pen_cap: float = -1.0,
    step_pen_per_nm: float = -1.0,
    out_of_zone_terminate: bool = False,
    out_of_zone_max_consecutive: int = 0,
    per_star_sr_scale: float = -1.0,
    loop_penalty: float = -1.0,
    run_name: str = '',                 # '' → auto-allocated run_N
    no_wandb: bool = False,
):
    if run_name:
        name = run_name
    else:
        run_n = next_ppo_run_number(prefix='run')
        name = f"run_{run_n}"
    print(f"  → PPO run: {name}")

    actor_relpath = f"runs/{bc_config}/run_{actor_run}/best.pt"
    print(f"  → seed actor: /{actor_relpath} (must already be on the volume)")

    args = {
        'run_name': name,
        'actor_ckpt_relpath': actor_relpath,
        'use_wandb': not no_wandb,
    }
    # Forward only the explicitly-set overrides (zero / sentinel = "leave default").
    if n_iters > 0:           args['n_iters'] = n_iters
    if n_rollouts > 0:        args['n_rollouts_per_iter'] = n_rollouts
    if n_workers > 0:         args['n_workers'] = n_workers
    if n_epochs > 0:          args['n_epochs'] = n_epochs
    if lr_actor > 0:          args['lr_actor'] = lr_actor
    if lr_critic > 0:         args['lr_critic'] = lr_critic
    if clip_eps > 0:          args['clip_epsilon'] = clip_eps
    if gamma > 0:             args['gamma'] = gamma
    if gae_lambda > 0:        args['gae_lambda'] = gae_lambda
    if target_kl > 0:         args['target_kl'] = target_kl
    if entropy_coef >= 0:     args['entropy_coef'] = entropy_coef
    if batch_size > 0:        args['batch_size'] = batch_size
    if max_steps_1_2 > 0:     args['max_timesteps_star_1_2'] = max_steps_1_2
    if max_steps_3 > 0:       args['max_timesteps_star_3'] = max_steps_3
    if seed:                  args['seed'] = seed
    # In-run eval.
    if eval_every > 0:        args['eval_every'] = eval_every
    if eval_cases > 0:        args['eval_cases'] = eval_cases
    # Reward-shaping knobs (0.0 is valid → forward on >= 0).
    if everywhere_pen >= 0:               args['everywhere_step_penalty'] = everywhere_pen
    if step_pen_cap >= 0:                 args['step_penalty_cap'] = step_pen_cap
    if step_pen_per_nm >= 0:              args['step_penalty_per_nm'] = step_pen_per_nm
    if out_of_zone_terminate:             args['out_of_zone_terminate'] = True
    if out_of_zone_max_consecutive > 0:   args['out_of_zone_max_consecutive'] = out_of_zone_max_consecutive
    if per_star_sr_scale >= 0:            args['per_star_sr_scale'] = per_star_sr_scale
    if loop_penalty >= 0:                 args['loop_penalty_per_step'] = loop_penalty

    print(f"  → dispatching to Modal (cpu=16, autoworkers=8). "
          f"Override `cpu=` in modal_config.py if you want more parallelism.")
    summary = train_ppo_remote.remote(args)
    print(f"\nremote summary: {summary}")

    print(f"\n  → pulling /runs_ppo/{name} back to rl_ppo/runs/{name}/ ...")
    local_dir = pull_ppo_run_back(name)
    print(f"  ✓ ready: {local_dir}")
