"""Local entrypoint for ONE continuous-run BLOCK on Modal.

The actual remote function `train_block_and_eval` lives in
`rl_ppo.modal_config` so it imports cleanly inside the Modal container
(modal mounts the entry script as `/root/<file>`, so module-level
`from rl_ppo...` imports here would fail — they're done lazily inside
the @app.function instead).

Workflow (driven manually by the trainer-in-the-loop):

    # Block 1 — iters 0..10 from BC seed
    PYTHONIOENCODING=utf-8 modal run rl_ppo/modal_continuous.py \\
        --run-name my_run \\
        --block-from-iter 0 --block-to-iter 10 \\
        --gamma 0.999 --everywhere-pen 0.001 \\
        --out-of-zone-terminate --out-of-zone-max-consecutive 10 \\
        --lr-actor 0.000001 --lr-critic 0.000003 \\
        --target-kl 0.01 --entropy-coef 0.10

    # Inspect rl_ppo/runs/<my_run>/iter_0010_eval/eval_metrics.json
    # then pick next-block knobs from the README's metric-to-knob table.

    # Block 2 — resume from iter_0010.pt, train to iter 20
    PYTHONIOENCODING=utf-8 modal run rl_ppo/modal_continuous.py \\
        --run-name my_run \\
        --block-from-iter 10 --block-to-iter 20 \\
        --gamma 0.999 --everywhere-pen 0.001 \\
        --out-of-zone-terminate --out-of-zone-max-consecutive 10 \\
        --per-star-sr-scale 0.5 \\
        --lr-actor 0.000001 --lr-critic 0.000003 \\
        --target-kl 0.01 --entropy-coef 0.10

Outputs on the volume / pulled to `rl_ppo/runs/<run-name>/`:

    iter_<to>.pt                          # ckpt at end of this block
    iter_<to>_eval/eval_metrics.json      # eval result for this block
    iter_<to>_eval/rollouts.csv           # per-episode rows
    iter_<to>_eval/trajectories.npz       # per-step (a, c)
    config_block_start_iter_<from>.json   # PPOConfig + reward state used
    log.jsonl                             # per-iter training log (appended)
"""
from __future__ import annotations

from rl_ppo.modal_config import (
    app,
    pull_ppo_run_back,
    train_block_and_eval,
)


@app.local_entrypoint()
def main(
    run_name: str,
    block_from_iter: int = 0,
    block_to_iter: int = 30,
    # Always-explicit reward knobs (per-block).
    gamma: float = 0.999,
    everywhere_pen: float = 0.001,
    step_pen_cap: float = 0.005,
    step_pen_per_nm: float = 0.0005,
    early_zone_multiplier: float = 1.0,
    early_window_steps: int = 300,
    early_drift_penalty: float = 0.0,
    clean_terminal_threshold: float = 0.0,
    drifty_success_value: float = 0.0,
    out_of_zone_terminate: bool = False,
    out_of_zone_max_consecutive: int = 5,
    per_star_sr_scale: float = 0.0,
    # Always-on base toggles (ablation only; default on).
    heading_intercept: bool = True,
    turn_final: bool = True,
    # Post-hoc loop penalty. 0 disables; ladder {0.02, 0.05, 0.10, 0.20}.
    loop_penalty: float = 0.0,
    # Common per-block overrides.
    n_rollouts: int = 512,
    entropy_coef: float = 0.02,
    target_kl: float = 0.02,
    lr_actor: float = 1e-5,
    lr_critic: float = 3e-5,
    eval_cases: int = 200,
    # In-run eval cadence WITHIN a block (0 = only the block-end eval).
    # Lets a single large block (e.g. all of Phase 1, 0->100) still emit
    # the per-10-iter eval curve without being chopped into 10-iter blocks.
    eval_every: int = 0,
    # BC seed (used only when block_from_iter=0).
    bc_config: str = 'bc_gmm_single_full',
    actor_run: int = 11,
    # Misc.
    seed: int = 0,
    no_wandb: bool = False,
):
    if not run_name:
        raise ValueError('--run-name is required')

    actor_relpath = f'runs/{bc_config}/run_{actor_run}/best.pt'
    args = {
        'run_name': run_name,
        'block_from_iter': block_from_iter,
        'block_to_iter': block_to_iter,
        'actor_ckpt_relpath': actor_relpath,
        'gamma': gamma,
        'everywhere_step_penalty': everywhere_pen,
        'step_penalty_cap': step_pen_cap,
        'step_penalty_per_nm': step_pen_per_nm,
        'early_zone_multiplier': early_zone_multiplier,
        'early_window_steps': early_window_steps,
        'early_drift_penalty': early_drift_penalty,
        'clean_terminal_threshold': clean_terminal_threshold,
        'drifty_success_value': drifty_success_value,
        'out_of_zone_terminate': out_of_zone_terminate,
        'out_of_zone_max_consecutive': out_of_zone_max_consecutive,
        'per_star_sr_scale': per_star_sr_scale,
        'heading_intercept_enabled': heading_intercept,
        'turn_final_enabled': turn_final,
        'loop_penalty_per_step': loop_penalty,
        'n_rollouts_per_iter': n_rollouts,
        'entropy_coef': entropy_coef,
        'target_kl': target_kl,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'eval_cases': eval_cases,
        'eval_every': eval_every,
        'seed': seed,
        'use_wandb': not no_wandb,
        'wandb_group': 'ppo_continuous',
    }
    print(f"  -- continuous run: {run_name}  block {block_from_iter}->{block_to_iter}")
    print(f"  -- gamma={gamma}  everywhere={everywhere_pen}  slope={step_pen_per_nm}  cap={step_pen_cap}  loop_pen={loop_penalty}")
    print(f"  -- early_zone_mult={early_zone_multiplier} x first {early_window_steps} policy steps")
    print(f"  -- early_drift_penalty={early_drift_penalty} per out-of-zone step in window (STAR-1/2 only)")
    print(f"  -- clean_terminal_threshold={clean_terminal_threshold} drifty_success_value={drifty_success_value}")
    print(f"  -- out_of_zone_terminate={out_of_zone_terminate} max_consecutive={out_of_zone_max_consecutive}")
    print(f"  -- per_star_sr_scale={per_star_sr_scale}")
    print(f"  -- rollouts/iter={n_rollouts}  eval_cases={eval_cases}/STAR")
    if block_from_iter == 0:
        print(f"  -- seed actor: /{actor_relpath} (BC seed)")
    else:
        print(f"  -- resume: /runs_ppo/{run_name}/iter_{block_from_iter:04d}.pt")

    result = train_block_and_eval.remote(args)
    print(f"\nremote summary: train={result['train_summary']}  "
          f"eval={result['eval_metrics_overall']}")

    print(f"\n  -- pulling /runs_ppo/{run_name} back to rl_ppo/runs/{run_name}/ ...")
    local_dir = pull_ppo_run_back(run_name)
    print(f"  [ok] ready: {local_dir}")
    eval_json = local_dir / f'iter_{block_to_iter:04d}_eval' / 'eval_metrics.json'
    if eval_json.exists():
        print(f"  [ok] eval metrics: {eval_json}")

    # Render successful/unsuccessful eval-traj PNGs LOCALLY (matplotlib
    # not in the Modal PPO image). One pair per block, lives alongside
    # the eval's rollouts.csv/trajectories.npz/eval_metrics.json.
    try:
        import sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[1]
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        import numpy as np
        from rl_bc.eval.viz_trajectories import render_from_results
        eval_dir = local_dir / f'iter_{block_to_iter:04d}_eval'
        npz_path = eval_dir / 'trajectories.npz'
        if npz_path.exists():
            z = np.load(npz_path, allow_pickle=False)
            stars = z['stars']; outcomes = z['outcomes']
            lens = z['lengths']
            a_concat = z['a_concat']; c_concat = z['c_concat']
            off = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
            results = []
            for i in range(len(stars)):
                results.append({
                    'star':    str(stars[i]),
                    'outcome': str(outcomes[i]),
                    'a_traj':  np.asarray(a_concat[off[i]:off[i + 1]],
                                          dtype=np.float32),
                    'c_traj':  np.asarray(c_concat[off[i]:off[i + 1]],
                                          dtype=np.float32),
                })
            n_succ, n_fail = render_from_results(results, eval_dir,
                                                 lim_nm=30.0)
            print(f"  [ok] rendered traj PNGs: succ={n_succ} fail={n_fail}")
    except Exception as exc:
        print(f"  [warn] PNG render failed: {type(exc).__name__}: {exc}")
