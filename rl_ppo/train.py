"""PPO training loop.

Per iteration:
  1. Broadcast the latest policy state_dict to all rollout workers.
  2. Collect `n_rollouts_per_iter` episodes in parallel.
  3. Compute GAE-λ advantages + returns per episode, flatten to one batch.
  4. Optionally normalize advantages.
  5. Run `n_epochs` over the batch with random minibatches; per minibatch:
       - policy loss = -mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))
       - value  loss = 0.5·mean((V_θ(s) − R)²)
       - entropy bonus = -ent_coef · mean(entropy_estimate)
       - clip grad-norm, step both optimizers
       - early-stop the epoch if approx-KL exceeds 1.5·target_kl
  6. Log metrics, save ckpt every `save_every` iters.

Usage:
    python -m rl_ppo.train
    python -m rl_ppo.train --n-iters 10 --n-rollouts 24 --n-workers 4

Most knobs live in `PPOConfig` (rl_ppo/config.py); the CLI exposes the
small subset useful per-invocation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from . import reward_zones as _reward_zones
from .config import PPOConfig
from .gae import compute_gae
from .policy import PPOPolicy
from .rollout import Trajectory, collect_rollouts, make_rollout_pool


# --------------------------------------------------------------------------- #
# Batch flattening
# --------------------------------------------------------------------------- #


def flatten_trajectories(trajectories: list[Trajectory],
                         gamma: float, lam: float
                         ) -> dict[str, torch.Tensor]:
    """Compute GAE per episode, then concatenate everything into flat
    tensors ready for minibatch SGD."""
    obs_chunks, act_chunks, lp_chunks = [], [], []
    adv_chunks, ret_chunks, val_chunks = [], [], []
    for traj in trajectories:
        adv, ret = compute_gae(
            rewards=list(traj.rewards),
            values=list(traj.values),
            dones=list(traj.dones),
            gamma=gamma, lam=lam,
        )
        obs_chunks.append(traj.obs)
        act_chunks.append(traj.actions)
        lp_chunks.append(traj.log_probs)
        adv_chunks.append(adv)
        ret_chunks.append(ret)
        val_chunks.append(traj.values)
    if not obs_chunks:
        raise RuntimeError("no usable trajectories in this batch")
    return {
        'obs': torch.from_numpy(np.concatenate(obs_chunks).astype(np.float32)),
        'actions': torch.from_numpy(np.concatenate(act_chunks).astype(np.float32)),
        'log_probs_old': torch.from_numpy(np.concatenate(lp_chunks).astype(np.float32)),
        'advantages': torch.from_numpy(np.concatenate(adv_chunks).astype(np.float32)),
        'returns': torch.from_numpy(np.concatenate(ret_chunks).astype(np.float32)),
        'values_old': torch.from_numpy(np.concatenate(val_chunks).astype(np.float32)),
    }


# --------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------- #


def ppo_update(policy: PPOPolicy,
               actor_opt: torch.optim.Optimizer,
               critic_opt: torch.optim.Optimizer,
               batch: dict[str, torch.Tensor],
               cfg: PPOConfig,
               device: torch.device) -> dict:
    """One PPO update (n_epochs * minibatches over the batch).

    Returns aggregate metrics (mean policy_loss, value_loss, entropy,
    approx_kl, clip_fraction).
    """
    obs = batch['obs'].to(device)
    actions = batch['actions'].to(device)
    log_probs_old = batch['log_probs_old'].to(device)
    advantages = batch['advantages'].to(device)
    returns = batch['returns'].to(device)

    if cfg.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = obs.shape[0]
    idx = np.arange(N)

    metrics = {
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'approx_kl': [], 'clip_fraction': [],
    }

    for epoch in range(cfg.n_epochs):
        np.random.shuffle(idx)
        epoch_kl_sum, epoch_kl_n = 0.0, 0
        early_stop = False
        for start in range(0, N, cfg.batch_size):
            mb = idx[start:start + cfg.batch_size]
            mb_t = torch.from_numpy(mb).to(device)
            mb_obs = obs.index_select(0, mb_t)
            mb_act = actions.index_select(0, mb_t)
            mb_lp_old = log_probs_old.index_select(0, mb_t)
            mb_adv = advantages.index_select(0, mb_t)
            mb_ret = returns.index_select(0, mb_t)

            out = policy.evaluate_actions(mb_obs, mb_act)
            log_probs_new = out['log_prob']
            entropy = out['entropy']
            value_pred = out['value']

            log_ratio = log_probs_new - mb_lp_old
            ratio = log_ratio.exp()

            # PPO clipped surrogate objective.
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon,
                                1.0 + cfg.clip_epsilon) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (value_pred - mb_ret).pow(2).mean()
            ent_bonus = entropy.mean()

            loss = (policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * ent_bonus)

            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            actor_params = list(policy.actor.parameters())
            critic_params = list(policy.critic.parameters())
            nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            actor_opt.step()
            critic_opt.step()

            with torch.no_grad():
                # Schulman's stable approximation to KL(π_old || π_new).
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item()

            metrics['policy_loss'].append(policy_loss.item())
            metrics['value_loss'].append(value_loss.item())
            metrics['entropy'].append(ent_bonus.item())
            metrics['approx_kl'].append(approx_kl)
            metrics['clip_fraction'].append(clip_frac)

            epoch_kl_sum += approx_kl
            epoch_kl_n += 1
            if cfg.target_kl is not None and epoch_kl_n > 0:
                mean_kl = epoch_kl_sum / epoch_kl_n
                if mean_kl > 1.5 * cfg.target_kl:
                    early_stop = True
                    break
        if early_stop:
            break

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# --------------------------------------------------------------------------- #
# GMM actor + value-net health check (run once per iter on the rollout batch)
# --------------------------------------------------------------------------- #


def gmm_health_check(policy: PPOPolicy, obs: torch.Tensor) -> dict:
    """Snapshot the GMM actor's per-state distribution stats.

    Catches the two failure modes we've seen on this dataset:
      1. Mixture collapse — one component dominates almost every state
         (mean_max_pi → 1, mean_entropy → 0). The softplus reparameterization
         can't prevent this; the row-entropy regularizer (BC-side) and PPO
         entropy_coef are the levers. Worth watching across PPO iters.
      2. σ / κ pinned at the structural bound — the floor/ceiling can't be
         violated by construction, but parameters parking AT the bound mean
         NLL wants tighter; the bound is the only thing holding it. If
         `at_sigma_floor_frac → 1` you have no per-state exploration noise
         headroom (PPO actor still works because of the bound, but any
         intermediate "trust the data variance" behavior is gone).

    All metrics computed under no_grad; ~50ms on a 30K-state batch.
    """
    from rl_bc.bc_gmm.model import LOG_STD_FLOOR, LOG_KAPPA_CEILING
    import math

    with torch.no_grad():
        c = policy.actor.encode(obs)
        (logits, _mu, kappa, _alt_mean, alt_log_std,
         _spd_mean, spd_log_std) = policy.actor.gmm_params(c)

        # Mixture-weight stats.
        pi = torch.softmax(logits, dim=-1)
        max_pi = pi.max(dim=-1).values
        H = -(pi.clamp_min(1e-8) * pi.clamp_min(1e-8).log()).sum(-1)
        dominant = pi.argmax(dim=-1)
        K = policy.actor.K
        dom_counts = torch.bincount(dominant, minlength=K).float()
        dom_fracs = (dom_counts / max(1, dominant.numel())).tolist()

        # σ / κ saturation: "at the bound" defined as within 1% of the
        # bound in log-space — empirically distinguishes "softplus is
        # active" from "model chose this value freely".
        FLOOR_EPS = 0.05    # log-space ~5% margin
        at_sigma_alt_floor = ((alt_log_std - LOG_STD_FLOOR).abs() < FLOOR_EPS).float().mean()
        at_sigma_spd_floor = ((spd_log_std - LOG_STD_FLOOR).abs() < FLOOR_EPS).float().mean()
        log_kappa = kappa.log()
        at_kappa_ceiling = ((log_kappa - LOG_KAPPA_CEILING).abs() < FLOOR_EPS).float().mean()

        # Distribution shape — useful to spot degeneracy.
        sigma_alt = alt_log_std.exp()
        sigma_spd = spd_log_std.exp()

        # Value head sanity.
        value = policy.critic(obs)

    metrics = {
        # Mixture collapse
        'actor/mean_max_pi': float(max_pi.mean().item()),
        'actor/mean_entropy': float(H.mean().item()),
        'actor/max_entropy_possible': math.log(K),
        'actor/dominant_comp_concentration': float(max(dom_fracs)),
        # σ / κ — median / tail / boundary fraction
        'actor/sigma_alt_median': float(sigma_alt.median().item()),
        'actor/sigma_alt_p95': float(sigma_alt.flatten().quantile(0.95).item()),
        'actor/sigma_spd_median': float(sigma_spd.median().item()),
        'actor/sigma_spd_p95': float(sigma_spd.flatten().quantile(0.95).item()),
        'actor/kappa_median': float(kappa.median().item()),
        'actor/kappa_p5': float(kappa.flatten().quantile(0.05).item()),
        'actor/at_sigma_alt_floor_frac': float(at_sigma_alt_floor.item()),
        'actor/at_sigma_spd_floor_frac': float(at_sigma_spd_floor.item()),
        'actor/at_kappa_ceiling_frac': float(at_kappa_ceiling.item()),
        # Value head
        'critic/value_mean': float(value.mean().item()),
        'critic/value_std': float(value.std().item()),
    }
    # Per-component dominance fraction — one series per component so we can
    # see e.g. "comp 2 ate 90% of states".
    for k, f in enumerate(dom_fracs):
        metrics[f'actor/comp_{k}_dominance_frac'] = float(f)
    return metrics


# --------------------------------------------------------------------------- #
# Per-iteration summary
# --------------------------------------------------------------------------- #


def summarize_trajectories(trajectories: list[Trajectory]) -> dict:
    """Outcomes / steps / LOC-interception geometry, aggregated per STAR + overall.

    The geometry block (`d_thr_nm`, `altitude_ft`, `gs_margin_ft`) is critical
    for sanity-checking the SUCCESS rate. If `d_thr_nm` median ≫ 8, the BC
    policy is "succeeding" by drive-by centerline crossings 15+ nm out, not
    by actual approach setups — the sim flips loc_intercepted on geometric
    crossings without distance gating.
    """
    from collections import Counter
    import numpy as np

    per_star: dict[str, dict] = {}
    geom_success: dict[str, list[tuple[float, float, float]]] = {}
    geom_loc_high: dict[str, list[tuple[float, float, float]]] = {}

    for t in trajectories:
        b = per_star.setdefault(t.star, Counter())
        b[t.outcome] += 1
        b['_steps'] += t.steps
        b['_n'] += 1
        # Reward accounting — sum of all per-step + terminal rewards
        # for this trajectory (numpy array on Trajectory.rewards).
        b['_R_sum'] += float(t.rewards.sum()) if t.rewards.size else 0.0
        if t.outcome in ('LOC_BELOW_GS', 'LOC_ABOVE_GS') and t.d_thr_nm is not None:
            bucket = (geom_success if t.outcome == 'LOC_BELOW_GS'
                      else geom_loc_high)
            bucket.setdefault(t.star, []).append(
                (float(t.d_thr_nm), float(t.altitude_ft),
                 float(t.altitude_ft) - float(t.gs_alt_ft))   # gs_margin
            )

    def _geom_stats(rows: list[tuple]) -> dict:
        if not rows:
            return {}
        arr = np.asarray(rows, dtype=np.float32)   # (N, 3): d, alt, gs_margin
        return {
            'd_thr_median': float(np.median(arr[:, 0])),
            'd_thr_p25': float(np.percentile(arr[:, 0], 25)),
            'd_thr_p75': float(np.percentile(arr[:, 0], 75)),
            'altitude_median': float(np.median(arr[:, 1])),
            'gs_margin_median': float(np.median(arr[:, 2])),  # negative = below GS
        }

    summary = {}
    total_n = 0
    total_succ = 0
    total_steps = 0
    total_R_succ = 0.0
    total_R_fail = 0.0
    n_R_succ = 0
    n_R_fail = 0
    all_success_geom = []
    all_lochigh_geom = []
    for star, b in per_star.items():
        n = b['_n']
        succ = b.get('LOC_BELOW_GS', 0)
        star_block = {
            'n': n,
            'success': succ,
            'loc_below_gs': succ,
            'loc_above_gs': b.get('LOC_ABOVE_GS', 0),
            'loc_behind_thr': b.get('LOC_BEHIND_THR', 0),
            'timeout': b.get('TIMEOUT', 0),
            'improper_exit': b.get('IMPROPER_EXIT', 0),
            'crashed': b.get('CRASHED', 0),
            'success_rate': succ / max(1, n),
            'mean_steps': b['_steps'] / max(1, n),
            'mean_reward': b['_R_sum'] / max(1, n),
        }
        s_geom = _geom_stats(geom_success.get(star, []))
        for k, v in s_geom.items():
            star_block[f'success_{k}'] = v
        lh_geom = _geom_stats(geom_loc_high.get(star, []))
        for k, v in lh_geom.items():
            star_block[f'loc_above_gs_{k}'] = v
        summary[star] = star_block
        total_n += n; total_succ += succ; total_steps += b['_steps']
        all_success_geom.extend(geom_success.get(star, []))
        all_lochigh_geom.extend(geom_loc_high.get(star, []))

    # Per-outcome reward bookkeeping (success vs failure).
    for t in trajectories:
        R = float(t.rewards.sum()) if t.rewards.size else 0.0
        if t.outcome == 'LOC_BELOW_GS':
            total_R_succ += R; n_R_succ += 1
        else:
            total_R_fail += R; n_R_fail += 1

    # Loop-penalty bookkeeping. `loop_rate` = fraction of trajs in this
    # iter where the detector flagged ANY looping (regardless of whether
    # the penalty was on); useful even when the penalty is 0 to track
    # whether the policy is producing more / fewer loops over training.
    # `mean_loop_penalty` is per-trajectory; only non-zero when the
    # penalty knob is on.
    n_loopers = sum(1 for t in trajectories if t.n_loop_steps > 0)
    total_loop_steps = sum(t.n_loop_steps for t in trajectories)
    total_penalized = sum(t.n_penalized for t in trajectories)
    total_loop_penalty = sum(t.loop_penalty_total for t in trajectories)

    overall = {
        'n': total_n,
        'success': total_succ,
        'success_rate': total_succ / max(1, total_n),
        'mean_steps': total_steps / max(1, total_n),
        'mean_reward_success': total_R_succ / max(1, n_R_succ),
        'mean_reward_failure': total_R_fail / max(1, n_R_fail),
        'mean_reward_all': (total_R_succ + total_R_fail) / max(1, total_n),
        'reward_separation': (total_R_succ / max(1, n_R_succ)
                              - total_R_fail / max(1, n_R_fail)),
        'loop_rate': n_loopers / max(1, total_n),
        'mean_loop_steps': total_loop_steps / max(1, total_n),
        'mean_loop_penalty': total_loop_penalty / max(1, total_n),
        'total_loop_penalty': float(total_loop_penalty),
        'total_loop_steps': int(total_loop_steps),
        'total_penalized_steps': int(total_penalized),
    }
    for k, v in _geom_stats(all_success_geom).items():
        overall[f'success_{k}'] = v
    for k, v in _geom_stats(all_lochigh_geom).items():
        overall[f'loc_above_gs_{k}'] = v
    summary['overall'] = overall
    return summary


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def train(cfg: PPOConfig, metric_hook: Callable[[dict], None] | None = None,
          resume_ckpt: str | Path | None = None,
          start_iter: int = 0,
          ) -> dict:
    """Train PPO. Optional resume from an existing PPO checkpoint.

    Args:
        cfg          : full PPOConfig (reward knobs + PPO hyperparams).
        metric_hook  : optional per-iter dict consumer (W&B etc.).
        resume_ckpt  : path to a `iter_NNNN.pt` (or `best.pt`) saved by
                       this trainer. Loads actor + critic + both
                       optimizer states. None = train from BC seed.
        start_iter   : absolute iter counter offset. With start_iter=30
                       and cfg.n_iters=20, this block writes iters
                       31..50 to log.jsonl and saves checkpoints with
                       absolute iter numbers (iter_0050.pt, etc.).
    """
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed + start_iter)  # nudge seed across blocks
    np.random.seed(cfg.seed + start_iter)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # ----- Install RUNTIME reward overrides from cfg into reward_zones.
    # Done before env.py is touched (env.py reads
    # `_reward_zones.EVERYWHERE_STEP_PENALTY` lazily and `step_reward`
    # reads STEP_PENALTY_CAP via its own module globals — both pick up
    # the override). After this point, cfg's reward knobs are
    # authoritative for the rest of the training block.
    # Load per-STAR recent SR from the prior eval (if it exists) so the
    # per-STAR scaling mechanism has real numbers to amplify off of.
    per_star_recent_sr = None
    if cfg.per_star_sr_scale > 0.0 and start_iter > 0:
        prior_eval = cfg.run_dir / f'iter_{start_iter:04d}_eval' / 'eval_metrics.json'
        if prior_eval.exists():
            with prior_eval.open() as f:
                m_prev = json.load(f)
            per_star_recent_sr = {
                star: float(m_prev['per_star'][star]['success_rate'])
                for star in cfg.stars
                if star in m_prev.get('per_star', {})
            }
            print(f"[ppo] per-STAR recent SR from {prior_eval.name}: "
                  f"{per_star_recent_sr}", flush=True)

    _reward_zones.set_runtime_overrides(
        everywhere_step_penalty=cfg.everywhere_step_penalty,
        step_penalty_cap=cfg.step_penalty_cap,
        step_penalty_per_nm=cfg.step_penalty_per_nm,
        early_zone_multiplier=cfg.early_zone_multiplier,
        early_window_steps=cfg.early_window_steps,
        early_drift_penalty=cfg.early_drift_penalty,
        clean_terminal_threshold=cfg.clean_terminal_threshold,
        drifty_success_value=cfg.drifty_success_value,
        out_of_zone_terminate=cfg.out_of_zone_terminate,
        out_of_zone_max_consecutive=cfg.out_of_zone_max_consecutive,
        per_star_sr_scale=cfg.per_star_sr_scale,
        per_star_recent_sr=per_star_recent_sr,
        heading_intercept_enabled=cfg.heading_intercept_enabled,
        turn_final_enabled=cfg.turn_final_enabled,
    )

    # Snapshot the FULL active reward state alongside the (resume-aware)
    # config so this block's run dir is self-describing — no hidden
    # module constants to chase later.
    block_label = f'block_start_iter_{start_iter:04d}'
    (cfg.run_dir / f'config_{block_label}.json').write_text(
        json.dumps({
            'ppo_config': {k: str(v) if isinstance(v, Path) else v
                           for k, v in asdict(cfg).items()},
            'reward_state': _reward_zones.get_runtime_overrides(),
            'resume_ckpt': str(resume_ckpt) if resume_ckpt else None,
            'start_iter': int(start_iter),
            'n_iters_this_block': int(cfg.n_iters),
        }, indent=2)
    )
    # Also keep a `config.json` mirror for back-compat with eval_metrics
    # tooling and the BC analysis scripts that look for that filename.
    (cfg.run_dir / 'config.json').write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(cfg).items()}, indent=2)
    )
    log_path = cfg.run_dir / 'log.jsonl'
    log_file = log_path.open('a', encoding='utf-8')

    # Local W&B setup: only fires if the caller didn't already wire a hook
    # (Modal's launcher provides its own). All metrics flow through this
    # hook as a flat dict — see the per-iter logging block below.
    wandb_run = None
    if metric_hook is None and cfg.wandb:
        try:
            import wandb
            run_name = cfg.wandb_run_name or cfg.run_dir.name
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                group=cfg.wandb_group,
                config={k: (str(v) if isinstance(v, Path) else v)
                        for k, v in asdict(cfg).items()},
                reinit=True,
            )
            print(f"[ppo] W&B: project={cfg.wandb_project} run={run_name}",
                  flush=True)

            def _wandb_hook(metrics: dict):
                step = metrics.get('iter')
                payload = {k: v for k, v in metrics.items() if k != 'iter'}
                wandb.log(payload, step=step)
            metric_hook = _wandb_hook
        except ImportError:
            print("[ppo] WARNING: --wandb requested but `wandb` not installed; "
                  "skipping W&B logging.", flush=True)
        except Exception as exc:
            print(f"[ppo] WARNING: W&B init failed ({type(exc).__name__}: {exc}); "
                  "continuing without W&B.", flush=True)

    print(f"[ppo] device={device}  run_dir={cfg.run_dir}", flush=True)
    print(f"[ppo] actor_ckpt={cfg.actor_ckpt}", flush=True)
    print(f"[ppo] {cfg.n_iters} iters × {cfg.n_rollouts_per_iter} rollouts/iter",
          flush=True)
    print(f"[ppo] γ={cfg.gamma}  λ={cfg.gae_lambda}  ε={cfg.clip_epsilon}  "
          f"target_kl={cfg.target_kl}", flush=True)

    # Build the master policy on the main process.
    policy = PPOPolicy.from_bc_checkpoint(
        cfg.actor_ckpt,
        value_hidden=cfg.value_hidden,
        value_dropout=cfg.value_dropout,
        device=device,
    )
    # CRITICAL: set eval mode for both rollout and update so dropout
    # doesn't fire random masks. The BC actor was trained with
    # dropout=0.1; if dropout is on during evaluate_actions but off
    # during rollout (workers do _POLICY.eval()), log_prob_new vs
    # log_prob_old diverge purely from random masks, blowing up KL
    # to ~1.5+ regardless of lr_actor. We don't need stochastic
    # regularization at the policy level — the GMM's mixture +
    # vMF/Gaussian sampling already provides exploration.
    policy.eval()
    actor_opt = AdamW(policy.actor.parameters(),
                       lr=cfg.lr_actor, weight_decay=cfg.weight_decay)
    critic_opt = AdamW(policy.critic.parameters(),
                        lr=cfg.lr_critic, weight_decay=cfg.weight_decay)

    # Optional resume: load actor + critic + optimizer state from a
    # prior PPO checkpoint. This lets a long continuous run progress
    # block-by-block with different reward params per block.
    if resume_ckpt is not None:
        resume_path = Path(resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f'resume ckpt not found: {resume_path}')
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        policy.actor.load_state_dict(blob['actor_state'])
        policy.critic.load_state_dict(blob['critic_state'])
        # Optimizer states are only present in full `iter_NNNN.pt`
        # checkpoints, not in `best.pt` (which is the leaner inference
        # ckpt). Fall back gracefully if absent.
        if 'actor_opt_state' in blob:
            actor_opt.load_state_dict(blob['actor_opt_state'])
        if 'critic_opt_state' in blob:
            critic_opt.load_state_dict(blob['critic_opt_state'])
        print(f"[ppo] resumed from {resume_path}  "
              f"(prior iter={blob.get('iter')}, "
              f"prior_best_sr={blob.get('overall_success_rate', 'n/a')})",
              flush=True)

    cfg_dict = {
        'airport_name': cfg.airport_name,
        'runway': cfg.runway,
        'warmup_wpts': cfg.warmup_wpts,
        'max_timesteps_star_1_2': cfg.max_timesteps_star_1_2,
        'max_timesteps_star_3': cfg.max_timesteps_star_3,
        'success_reward': cfg.success_reward,
        'failure_reward': cfg.failure_reward,
        'gs_capture_buffer_ft': cfg.gs_capture_buffer_ft,
        'value_hidden': cfg.value_hidden,
        'value_dropout': cfg.value_dropout,
        # Loop-detector knobs forwarded to rollout workers so the
        # post-hoc per-step penalty is applied inside each worker.
        # All four are snapshotted in config.json via PPOConfig.
        'loop_penalty_per_step': cfg.loop_penalty_per_step,
        'loop_prox_radius_nm': cfg.loop_prox_radius_nm,
        'loop_min_gap_steps': cfg.loop_min_gap_steps,
        'loop_min_detour_nm': cfg.loop_min_detour_nm,
        # Reward-zone overrides forwarded to workers so `_init_worker`
        # can call set_runtime_overrides() inside each worker process.
        # Required because workers spawn fresh module imports that
        # don't inherit the main process's globals.
        'reward_everywhere_step_penalty': cfg.everywhere_step_penalty,
        'reward_step_penalty_cap': cfg.step_penalty_cap,
        'reward_step_penalty_per_nm': cfg.step_penalty_per_nm,
        'reward_early_zone_multiplier': cfg.early_zone_multiplier,
        'reward_early_window_steps': cfg.early_window_steps,
        'reward_early_drift_penalty': cfg.early_drift_penalty,
        'reward_clean_terminal_threshold': cfg.clean_terminal_threshold,
        'reward_drifty_success_value': cfg.drifty_success_value,
        'reward_out_of_zone_terminate': cfg.out_of_zone_terminate,
        'reward_out_of_zone_max_consecutive': cfg.out_of_zone_max_consecutive,
        'reward_per_star_sr_scale': cfg.per_star_sr_scale,
        'reward_per_star_recent_sr': per_star_recent_sr,  # dict or None
        'reward_heading_intercept_enabled': cfg.heading_intercept_enabled,
        'reward_turn_final_enabled': cfg.turn_final_enabled,
    }

    import os
    n_workers = cfg.resolve_n_workers()
    cpu_total = os.cpu_count() or 0
    t_pool_start = time.time()
    print(f"[ppo] spawning {n_workers} rollout workers "
          f"(cpu_count={cpu_total}, n_workers config={cfg.n_workers}, "
          f"autodetect ≈ cpu_count//2 to skip SMT)...", flush=True)
    pool = make_rollout_pool(cfg.actor_ckpt, cfg_dict, n_workers)
    print(f"[ppo] pool ready in {time.time() - t_pool_start:.1f}s", flush=True)

    seed_offset = cfg.seed * 1_000_000
    best_success_rate = -1.0

    try:
        for it_local in range(cfg.n_iters):
            # `it` is the ABSOLUTE iter counter (offset by start_iter)
            # so checkpoint filenames and log entries are continuous
            # across blocks. `it_local` is only used to govern the
            # loop length within this block.
            it = start_iter + it_local
            t_iter = time.time()
            # 1. Broadcast latest policy state to workers (via per-job arg).
            policy_state = policy.state_dict_split()
            # 2. Collect rollouts.
            trajectories = collect_rollouts(
                pool=pool, n_rollouts=cfg.n_rollouts_per_iter,
                stars=cfg.stars, seed_offset=seed_offset,
                policy_state=policy_state,
                verbose=(it % cfg.log_every == 0),
            )
            seed_offset += cfg.n_rollouts_per_iter * 7  # avoid seed reuse
            if not trajectories:
                print(f"[ppo] iter {it}: no trajectories — aborting", flush=True)
                break

            # 3. Flatten + GAE.
            batch = flatten_trajectories(trajectories,
                                          gamma=cfg.gamma, lam=cfg.gae_lambda)
            # 4. PPO update.
            update_metrics = ppo_update(policy, actor_opt, critic_opt,
                                         batch, cfg, device)

            # 5. Log.
            ep_summary = summarize_trajectories(trajectories)
            overall = ep_summary['overall']

            # Diagnostics on the batch itself — these are cheap and useful
            # for spotting bad-batch iterations early.
            ret = batch['returns'].numpy()
            val = batch['values_old'].numpy()
            adv = batch['advantages'].numpy()
            var_ret = float(ret.var())
            explained_var = float(1.0 - (ret - val).var() / (var_ret + 1e-8))
            batch_stats = {
                'returns/mean': float(ret.mean()),
                'returns/std': float(ret.std()),
                'returns/min': float(ret.min()),
                'returns/max': float(ret.max()),
                'advantages/mean': float(adv.mean()),
                'advantages/std': float(adv.std()),
                'value/explained_variance': explained_var,
            }

            # GMM actor health (snapshot of POST-update policy on the batch).
            health = gmm_health_check(policy, batch['obs'].to(device))

            sec = time.time() - t_iter
            # Flat metrics — every numeric goes here, W&B-ready. per-STAR is
            # flattened into `per_star/<NAME>/<key>` keys so each STAR gets
            # its own time-series plot.
            flat = {
                'iter': it,
                'time_sec': sec,
                'n_rollouts': len(trajectories),
                'n_steps': int(batch['obs'].shape[0]),
            }
            for k, v in update_metrics.items():
                flat[f'metric/{k}'] = v
            for k, v in overall.items():
                flat[f'rollout/{k}'] = v
            for k, v in batch_stats.items():
                flat[k] = v
            for k, v in health.items():
                flat[k] = v
            for star, star_d in ep_summary.items():
                if star == 'overall':
                    continue
                for sk, sv in star_d.items():
                    flat[f'per_star/{star}/{sk}'] = sv

            # JSONL keeps the nested structure for greppability + the
            # flat copy for parity with W&B.
            log_record = {**flat, 'per_star_nested': ep_summary}
            log_file.write(json.dumps(log_record) + '\n')
            log_file.flush()
            if metric_hook is not None:
                metric_hook(flat)

            if it_local % cfg.log_every == 0:
                # Pull the success-geometry stats if any successes happened
                # this iter. Median d_thr at SUCCESS is the key sanity check:
                # if it's ≫ 8 nm, BC is "winning" via geometric drive-bys.
                d_med = overall.get('success_d_thr_median', float('nan'))
                alt_med = overall.get('success_altitude_median', float('nan'))
                margin_med = overall.get('success_gs_margin_median', float('nan'))
                R_succ = overall.get('mean_reward_success', float('nan'))
                R_fail = overall.get('mean_reward_failure', float('nan'))
                R_sep  = overall.get('reward_separation',  float('nan'))
                print(f"[ppo] it {it:>4d}/{start_iter + cfg.n_iters}  "
                      f"sr={overall['success_rate']:.3f}  "
                      f"R_succ={R_succ:+6.1f}  R_fail={R_fail:+6.1f}  "
                      f"R_sep={R_sep:+5.1f}  "
                      f"steps/ep={overall['mean_steps']:.0f}  "
                      f"pi_loss={update_metrics['policy_loss']:+.4f}  "
                      f"v_loss={update_metrics['value_loss']:.4f}  "
                      f"kl={update_metrics['approx_kl']:.4f}  "
                      f"clip_frac={update_metrics['clip_fraction']:.3f}  "
                      f"ev={explained_var:+.3f}  "
                      f"mix_H={health['actor/mean_entropy']:.2f}  "
                      f"| LOC_OK@ d={d_med:5.1f}nm alt={alt_med:5.0f}ft "
                      f"gs_margin={margin_med:+5.0f}ft  ({sec:.1f}s)", flush=True)

            # 6. Save. We persist actor + critic weights AND optimizer
            # states so PPO can be cleanly resumed without losing Adam's
            # momentum / second-moment buffers. `best.pt` is a leaner
            # checkpoint (no optimizer state — it's for inference / next
            # PPO seed, not for resuming this training run).
            # End-of-block save uses local iter so it fires regardless
            # of start_iter offset.
            if (it + 1) % cfg.save_every == 0 or it_local == cfg.n_iters - 1:
                ckpt_path = cfg.run_dir / f'iter_{it + 1:04d}.pt'
                torch.save({
                    'iter': it + 1,
                    'config': {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in asdict(cfg).items()},
                    'actor_state': policy.actor.state_dict(),
                    'critic_state': policy.critic.state_dict(),
                    'actor_opt_state': actor_opt.state_dict(),
                    'critic_opt_state': critic_opt.state_dict(),
                    'overall_success_rate': overall['success_rate'],
                    'best_success_rate_so_far': best_success_rate,
                }, ckpt_path)
                if overall['success_rate'] > best_success_rate:
                    best_success_rate = overall['success_rate']
                    torch.save({
                        'iter': it + 1,
                        'actor_state': policy.actor.state_dict(),
                        'critic_state': policy.critic.state_dict(),
                        'overall_success_rate': overall['success_rate'],
                    }, cfg.run_dir / 'best.pt')

            # Every 10 iters: SAVE this iter's per-trajectory data as
            # npz so we can render successful/unsuccessful PNGs locally
            # after pulling back (the Modal PPO image doesn't have
            # matplotlib, and we don't want to slow training to add it).
            if (it + 1) % 10 == 0:
                snap_dir = cfg.run_dir / 'trajectory_snapshots'
                snap_dir.mkdir(parents=True, exist_ok=True)
                stars, outcomes, lens = [], [], []
                a_chunks, c_chunks = [], []
                for t in trajectories:
                    if t.a_traj is None or len(t.a_traj) < 2:
                        continue
                    stars.append(t.star); outcomes.append(t.outcome)
                    lens.append(len(t.a_traj))
                    a_chunks.append(t.a_traj.astype(np.float32))
                    c_chunks.append(t.c_traj.astype(np.float32))
                if stars:
                    np.savez_compressed(
                        snap_dir / f'iter_{it + 1:04d}.npz',
                        stars=np.array(stars),
                        outcomes=np.array(outcomes),
                        lengths=np.array(lens, dtype=np.int32),
                        a_concat=np.concatenate(a_chunks),
                        c_concat=np.concatenate(c_chunks),
                    )

            # In-run periodic eval. For single straight `--n-iters` runs
            # this gives the per-K-iter eval curve (iter_NNNN_eval/, same
            # artifacts as the continuous driver) without paying the
            # per-block modal cold-start. The training pool stays open and
            # idle while run_eval spins its own; both fit on the box.
            if cfg.eval_every > 0 and (it + 1) % cfg.eval_every == 0:
                eval_ckpt = cfg.run_dir / f'iter_{it + 1:04d}.pt'
                if not eval_ckpt.exists():
                    # Eval point isn't a save point — persist a ckpt now.
                    torch.save({
                        'iter': it + 1,
                        'config': {k: (str(v) if isinstance(v, Path) else v)
                                   for k, v in asdict(cfg).items()},
                        'actor_state': policy.actor.state_dict(),
                        'critic_state': policy.critic.state_dict(),
                        'actor_opt_state': actor_opt.state_dict(),
                        'critic_opt_state': critic_opt.state_dict(),
                        'overall_success_rate': overall['success_rate'],
                        'best_success_rate_so_far': best_success_rate,
                    }, eval_ckpt)
                try:
                    from rl_ppo.eval_runner import run_eval
                    eval_out = cfg.run_dir / f'iter_{it + 1:04d}_eval'
                    em = run_eval(eval_ckpt, eval_out,
                                  cases=cfg.eval_cases,
                                  seed_base=10_000 + (it + 1) * 1000)
                    eo = em['overall']
                    print(f"[ppo] EVAL it {it + 1}: "
                          f"SR={eo['success_rate'] * 100:.1f}%  "
                          f"green={eo['macro_pct_steps_in_green'] * 100:.1f}%  "
                          f"in_range="
                          f"{eo['macro_pct_within_length_range'] * 100:.1f}%",
                          flush=True)
                    if metric_hook is not None:
                        metric_hook({
                            'iter': it + 1,
                            'eval/success_rate': eo['success_rate'],
                            'eval/macro_pct_steps_in_green':
                                eo['macro_pct_steps_in_green'],
                            'eval/macro_pct_within_length_range':
                                eo['macro_pct_within_length_range'],
                        })
                except Exception as exc:
                    print(f"[ppo] EVAL it {it + 1} FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
    finally:
        pool.close()
        pool.join()
        log_file.close()
        if wandb_run is not None:
            import wandb
            wandb.finish()

    print(f"[ppo] done. best_success_rate={best_success_rate:.3f}", flush=True)
    return {'best_success_rate': best_success_rate}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='PPO training on top of the BC GMM actor')
    p.add_argument('--actor-ckpt', type=str, default=None)
    p.add_argument('--run-dir', type=str, default=None)
    p.add_argument('--n-iters', type=int, default=None)
    p.add_argument('--n-rollouts', type=int, default=None)
    p.add_argument('--n-workers', type=int, default=None,
                   help='0 = autodetect from cpu_count // 2 (skip SMT siblings). '
                        'Explicit N pins to that count.')
    p.add_argument('--n-epochs', type=int, default=None)
    p.add_argument('--lr-actor', type=float, default=None)
    p.add_argument('--lr-critic', type=float, default=None)
    p.add_argument('--clip-eps', type=float, default=None)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--gae-lambda', type=float, default=None)
    p.add_argument('--target-kl', type=float, default=None)
    p.add_argument('--entropy-coef', type=float, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--wandb', action='store_true',
                   help='Log per-iter metrics to W&B (project=atc-ppo).')
    p.add_argument('--wandb-run', type=str, default=None,
                   help='W&B run name (default: run_dir basename).')
    p.add_argument('--wandb-project', type=str, default=None)
    # Continuous-run controls.
    p.add_argument('--resume', type=str, default=None,
                   help='Path to a previously-saved iter_NNNN.pt to '
                        'resume actor + critic + optimizer state from.')
    p.add_argument('--start-iter', type=int, default=0,
                   help='Absolute iter counter offset for this block. '
                        'With --start-iter 30 and --n-iters 20 the block '
                        'writes iters 31..50.')
    # Tunable reward knobs (per-block).
    p.add_argument('--everywhere-pen', type=float, default=None,
                   help='Per-step penalty applied EVERY step '
                        '(ladder: 0, 0.001, 0.002).')
    p.add_argument('--step-pen-cap', type=float, default=None,
                   help='Cap on out-of-zone step penalty '
                        '(ladder: 0.005, 0.010, 0.020, 0.040, 0.050).')
    p.add_argument('--step-pen-per-nm', type=float, default=None,
                   help='Slope of out-of-zone step penalty per nm '
                        '(ladder: 0.0005, 0.001, 0.002, 0.005).')
    p.add_argument('--early-zone-multiplier', type=float, default=None,
                   help='Multiplier on out-of-zone penalty during the '
                        'first --early-window-steps of each episode '
                        '(ladder: 1, 3, 5, 10).')
    p.add_argument('--early-window-steps', type=int, default=None,
                   help='Window length (policy steps) for the early-zone '
                        'multiplier (ladder: 100, 200, 300, 500).')
    p.add_argument('--early-drift-penalty', type=float, default=None,
                   help='FLAT per-out-of-zone-step penalty in early '
                        'window (STAR-1/2 only). Half-failure lever. '
                        'Ladder: 0, 0.02, 0.05, 0.10.')
    p.add_argument('--clean-terminal-threshold', type=float, default=None,
                   help='STAR-1/2 success only gets full +10 terminal '
                        'if frac_in_zone >= threshold (else terminal '
                        'becomes drifty_success_value). 1.0 = disabled. '
                        'Ladder: 1.0, 0.95, 0.9, 0.8, 0.7.')
    p.add_argument('--drifty-success-value', type=float, default=None,
                   help='Terminal value to use for "successful but '
                        'drifty" trajectories. Default 0.')
    p.add_argument('--out-of-zone-terminate', action='store_true',
                   help='Enable OUT_OF_ZONE early-termination for '
                        'STAR-1/2 family. Terminates episode as '
                        'failure (-10) after N consecutive out-of-zone '
                        'steps.')
    p.add_argument('--out-of-zone-max-consecutive', type=int, default=None,
                   help='Tolerance for OUT_OF_ZONE termination (default 5).')
    p.add_argument('--per-star-sr-scale', type=float, default=None,
                   help='Per-STAR multiplicative success-reward scaling. '
                        '0=disabled. 0.5=mild. terminal *= 1+scale*(1-recent_SR_star).')
    p.add_argument('--loop-penalty', type=float, default=None,
                   help='Per-step penalty applied retroactively to the '
                        'SECOND HALF of looping timesteps detected by '
                        'loop_detector.detect_looping. 0 disables. '
                        'Ladder: 0, 0.02, 0.05, 0.10.')
    p.add_argument('--loop-prox-radius-nm', type=float, default=None)
    p.add_argument('--loop-min-gap-steps', type=int, default=None)
    p.add_argument('--loop-min-detour-nm', type=float, default=None)
    return p


def main():
    args = _build_argparser().parse_args()
    cfg = PPOConfig()
    overrides = {
        'actor_ckpt': Path(args.actor_ckpt) if args.actor_ckpt else None,
        'run_dir': Path(args.run_dir) if args.run_dir else None,
        'n_iters': args.n_iters,
        'n_rollouts_per_iter': args.n_rollouts,
        'n_workers': args.n_workers,
        'n_epochs': args.n_epochs,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'clip_epsilon': args.clip_eps,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'target_kl': args.target_kl,
        'entropy_coef': args.entropy_coef,
        'batch_size': args.batch_size,
        'device': args.device,
        'seed': args.seed,
        'wandb_run_name': args.wandb_run,
        'wandb_project': args.wandb_project,
        'everywhere_step_penalty': args.everywhere_pen,
        'step_penalty_cap': args.step_pen_cap,
        'step_penalty_per_nm': args.step_pen_per_nm,
        'early_zone_multiplier': args.early_zone_multiplier,
        'early_window_steps': args.early_window_steps,
        'early_drift_penalty': args.early_drift_penalty,
        'clean_terminal_threshold': args.clean_terminal_threshold,
        'drifty_success_value': args.drifty_success_value,
        'out_of_zone_terminate': True if args.out_of_zone_terminate else None,
        'out_of_zone_max_consecutive': args.out_of_zone_max_consecutive,
        'per_star_sr_scale': args.per_star_sr_scale,
        'loop_penalty_per_step': args.loop_penalty,
        'loop_prox_radius_nm': args.loop_prox_radius_nm,
        'loop_min_gap_steps': args.loop_min_gap_steps,
        'loop_min_detour_nm': args.loop_min_detour_nm,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.wandb:
        cfg.wandb = True
    summary = train(
        cfg,
        resume_ckpt=args.resume,
        start_iter=int(args.start_iter),
    )
    print('summary:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
