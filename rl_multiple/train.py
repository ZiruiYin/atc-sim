"""Multi-plane PPO training loop (Phase 1).

Same structure as `rl_ppo.train` but trains the DeltaHead + Critic on
top of a FROZEN GMM. Reuses `rl_ppo.gae.compute_gae` and
`rl_ppo.reward_zones` directly (no need to duplicate).

Per iteration:
  1. Broadcast latest delta_head + critic state to rollout workers.
  2. Collect `n_rollouts_per_iter` episodes via MultiPPOEnv + CombinedPolicy.
  3. Compute GAE-λ on the full 79-D obs; flatten.
  4. PPO update — gradients flow into delta_head + critic only.
  5. Log + save.

CLI mirrors rl_ppo.train.
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
from torch.optim import AdamW

from rl_ppo import reward_zones as _reward_zones
from rl_ppo.gae import compute_gae

from rl_multiple.config import MultiPPOConfig
from rl_multiple.policy import CombinedPolicy
from rl_multiple.rollout import Trajectory, collect_rollouts, make_rollout_pool
from rl_multiple.runtime import resolve_bc_seed_path


# --------------------------------------------------------------------------- #
# Batch flattening (carries the full 79-D obs)
# --------------------------------------------------------------------------- #


def flatten_trajectories(trajectories: list[Trajectory],
                         gamma: float, lam: float
                         ) -> dict[str, torch.Tensor]:
    obs_chunks, act_chunks, lp_chunks = [], [], []
    adv_chunks, ret_chunks, val_chunks = [], [], []
    for traj in trajectories:
        adv, ret = compute_gae(
            rewards=list(traj.rewards),
            values=list(traj.values),
            dones=list(traj.dones),
            gamma=gamma, lam=lam,
        )
        obs_chunks.append(traj.obs)            # (T, 79)
        act_chunks.append(traj.actions)        # (T, 3) delta
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
# PPO update — same algorithm as rl_ppo.train, parameter sets are different.
# --------------------------------------------------------------------------- #


def ppo_update(policy: CombinedPolicy,
               actor_opt: torch.optim.Optimizer,
               critic_opt: torch.optim.Optimizer,
               batch: dict[str, torch.Tensor],
               cfg: MultiPPOConfig,
               device: torch.device) -> dict:
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

    for _epoch in range(cfg.n_epochs):
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
            actor_params = list(policy.delta_head.parameters())
            critic_params = list(policy.critic.parameters())
            nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            actor_opt.step()
            critic_opt.step()

            with torch.no_grad():
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
# Delta-head health check (cheap)
# --------------------------------------------------------------------------- #


def delta_head_health(policy: CombinedPolicy, obs: torch.Tensor) -> dict:
    """Per-state distribution stats for the delta head. Cheap; runs
    once per iter on the rollout batch."""
    with torch.no_grad():
        mu, sigma = policy.delta_head(obs)
        # mu is in physical units; the head's tanh clamps it to
        # ±(hdg, alt, spd) clamp values stored in mu_scale.
        scales = policy.delta_head.mu_scale.to(mu.device)
        sat_frac = ((mu.abs() / scales) > 0.95).float().mean(dim=0)
        value = policy.critic(obs)
    return {
        'delta/mu_hdg_mean':  float(mu[:, 0].mean().item()),
        'delta/mu_hdg_std':   float(mu[:, 0].std().item()),
        'delta/mu_alt_mean':  float(mu[:, 1].mean().item()),
        'delta/mu_alt_std':   float(mu[:, 1].std().item()),
        'delta/mu_spd_mean':  float(mu[:, 2].mean().item()),
        'delta/mu_spd_std':   float(mu[:, 2].std().item()),
        'delta/sigma_hdg_mean': float(sigma[:, 0].mean().item()),
        'delta/sigma_alt_mean': float(sigma[:, 1].mean().item()),
        'delta/sigma_spd_mean': float(sigma[:, 2].mean().item()),
        'delta/sat_hdg_frac': float(sat_frac[0].item()),
        'delta/sat_alt_frac': float(sat_frac[1].item()),
        'delta/sat_spd_frac': float(sat_frac[2].item()),
        'critic/value_mean':  float(value.mean().item()),
        'critic/value_std':   float(value.std().item()),
    }


# --------------------------------------------------------------------------- #
# Per-iteration summary (lifted from rl_ppo.train, identical shape)
# --------------------------------------------------------------------------- #


def summarize_trajectories(trajectories: list[Trajectory]) -> dict:
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
        b['_R_sum'] += float(t.rewards.sum()) if t.rewards.size else 0.0
        # Phase 2: track per-trajectory warning seconds (collision-warn
        # ticks). Default to 0 for Phase-1 Trajectory (no such field).
        b['_warn'] += int(getattr(t, 'warning_seconds', 0))
        if t.outcome in ('LOC_BELOW_GS', 'LOC_ABOVE_GS') and t.d_thr_nm is not None:
            bucket = (geom_success if t.outcome == 'LOC_BELOW_GS'
                      else geom_loc_high)
            bucket.setdefault(t.star, []).append(
                (float(t.d_thr_nm), float(t.altitude_ft),
                 float(t.altitude_ft) - float(t.gs_alt_ft))
            )

    def _geom_stats(rows: list[tuple]) -> dict:
        if not rows:
            return {}
        arr = np.asarray(rows, dtype=np.float32)
        return {
            'd_thr_median':  float(np.median(arr[:, 0])),
            'altitude_median': float(np.median(arr[:, 1])),
            'gs_margin_median': float(np.median(arr[:, 2])),
        }

    summary = {}
    total_n = 0; total_succ = 0; total_steps = 0
    total_warning_seconds = 0
    total_crashes = 0
    total_truncated = 0
    all_success_geom: list = []
    all_lochigh_geom: list = []
    for star, b in per_star.items():
        n = b['_n']
        # Phase 2 success = LANDED. Phase 1 success = LOC_BELOW_GS.
        # Count both so the same summarizer works for either mode.
        succ_phase2 = b.get('LANDED', 0)
        succ_phase1 = b.get('LOC_BELOW_GS', 0)
        succ = succ_phase2 + succ_phase1
        star_block = {
            'n': n, 'success': succ,
            'landed': succ_phase2,
            'loc_below_gs': succ_phase1,
            'loc_above_gs': b.get('LOC_ABOVE_GS', 0),
            'loc_behind_thr': b.get('LOC_BEHIND_THR', 0),
            'timeout': b.get('TIMEOUT', 0),
            'improper_exit': b.get('IMPROPER_EXIT', 0),
            'crashed': b.get('CRASHED', 0),
            'crash': b.get('CRASH', 0),
            'truncated': b.get('TRUNCATED', 0),
            'out_of_zone': b.get('OUT_OF_ZONE', 0),
            'success_rate': succ / max(1, n),
            'mean_steps': b['_steps'] / max(1, n),
            'mean_reward': b['_R_sum'] / max(1, n),
            'warning_seconds': b.get('_warn', 0),
            'mean_warning_seconds': b.get('_warn', 0) / max(1, n),
        }
        for k, v in _geom_stats(geom_success.get(star, [])).items():
            star_block[f'success_{k}'] = v
        for k, v in _geom_stats(geom_loc_high.get(star, [])).items():
            star_block[f'loc_above_gs_{k}'] = v
        summary[star] = star_block
        total_n += n; total_succ += succ; total_steps += b['_steps']
        total_warning_seconds += int(b.get('_warn', 0))
        total_crashes += int(b.get('CRASH', 0)) + int(b.get('CRASHED', 0))
        total_truncated += int(b.get('TRUNCATED', 0))
        all_success_geom.extend(geom_success.get(star, []))
        all_lochigh_geom.extend(geom_loc_high.get(star, []))

    # Phase-2 specific failure breakdown — distinguish CRASH (collision-
    # avoidance failure, the new objective) from POLICY failures (LOC
    # captures gone wrong, TIMEOUTs, exits — would indicate the radar
    # head broke Phase 1's landing competence).
    total_policy_fail = 0
    for star, b in per_star.items():
        total_policy_fail += int(b.get('LOC_ABOVE_GS', 0))
        total_policy_fail += int(b.get('LOC_BEHIND_THR', 0))
        total_policy_fail += int(b.get('TIMEOUT', 0))
        total_policy_fail += int(b.get('IMPROPER_EXIT', 0))

    overall = {
        'n': total_n, 'success': total_succ,
        'success_rate': total_succ / max(1, total_n),
        'mean_steps': total_steps / max(1, total_n),
        'warning_seconds_total': total_warning_seconds,
        'mean_warning_seconds_per_traj': total_warning_seconds / max(1, total_n),
        'n_crashes': total_crashes,
        'n_truncated': total_truncated,
        'n_policy_failures': total_policy_fail,
        'crash_rate': total_crashes / max(1, total_n),
        'policy_failure_rate': total_policy_fail / max(1, total_n),
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


def _resolve_bc_seed(cfg: MultiPPOConfig) -> Path:
    """Use cfg.bc_seed_ckpt if set; else resolve from the PPO ckpt's
    config.actor_ckpt field via the same helper the watch uses."""
    if cfg.bc_seed_ckpt and Path(cfg.bc_seed_ckpt).exists():
        return Path(cfg.bc_seed_ckpt)
    blob = torch.load(cfg.ppo_ckpt, map_location='cpu', weights_only=False)
    return resolve_bc_seed_path(blob)


def train(cfg: MultiPPOConfig,
          metric_hook: Callable[[dict], None] | None = None,
          resume_ckpt: str | Path | None = None,
          start_iter: int = 0,
          ) -> dict:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed + start_iter)
    np.random.seed(cfg.seed + start_iter)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # Reward overrides (identical to rl_ppo). per_star_recent_sr loaded
    # from prior eval if the mechanism is on.
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
    )

    block_label = f'block_start_iter_{start_iter:04d}'
    (cfg.run_dir / f'config_{block_label}.json').write_text(
        json.dumps({
            'multi_ppo_config': {k: str(v) if isinstance(v, Path) else v
                                  for k, v in asdict(cfg).items()},
            'reward_state': _reward_zones.get_runtime_overrides(),
            'resume_ckpt': str(resume_ckpt) if resume_ckpt else None,
            'start_iter': int(start_iter),
            'n_iters_this_block': int(cfg.n_iters),
        }, indent=2)
    )
    (cfg.run_dir / 'config.json').write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(cfg).items()}, indent=2)
    )
    log_path = cfg.run_dir / 'log.jsonl'
    log_file = log_path.open('a', encoding='utf-8')

    # Local W&B (mirror rl_ppo.train).
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

            def _wandb_hook(metrics: dict):
                step = metrics.get('iter')
                payload = {k: v for k, v in metrics.items() if k != 'iter'}
                wandb.log(payload, step=step)
            metric_hook = _wandb_hook
        except ImportError:
            print("[multi-ppo] wandb not installed; skipping", flush=True)
        except Exception as exc:
            print(f"[multi-ppo] wandb init failed: {exc}", flush=True)

    bc_seed = _resolve_bc_seed(cfg)
    print(f"[multi-ppo] device={device}  run_dir={cfg.run_dir}", flush=True)
    print(f"[multi-ppo] frozen-GMM ckpt: {cfg.ppo_ckpt}", flush=True)
    print(f"[multi-ppo] BC seed (arch):  {bc_seed}", flush=True)
    print(f"[multi-ppo] {cfg.n_iters} iters × {cfg.n_rollouts_per_iter} rollouts/iter",
          flush=True)
    print(f"[multi-ppo] γ={cfg.gamma}  λ={cfg.gae_lambda}  ε={cfg.clip_epsilon}  "
          f"target_kl={cfg.target_kl}", flush=True)

    # Build the master CombinedPolicy on the main process.
    policy = CombinedPolicy.from_ppo_ckpt(
        cfg.ppo_ckpt,
        bc_seed_path=bc_seed,
        density_n_bins=cfg.density_n_bins,
        delta_hidden=cfg.delta_hidden,
        value_hidden=cfg.value_hidden,
        value_dropout=cfg.value_dropout,
        delta_kwargs={
            'hdg_clamp_deg':  cfg.delta_hdg_clamp_deg,
            'alt_clamp_kft':  cfg.delta_alt_clamp_kft,
            'spd_clamp_kt':   cfg.delta_spd_clamp_kt,
            'log_sigma_init': cfg.delta_log_sigma_init,
            'log_sigma_min':  cfg.delta_log_sigma_min,
            'log_sigma_max':  cfg.delta_log_sigma_max,
        },
        device=device,
    )
    policy.eval()        # `.train()` is overridden to keep GMM in eval

    actor_opt = AdamW(policy.delta_head.parameters(),
                       lr=cfg.lr_actor, weight_decay=cfg.weight_decay)
    critic_opt = AdamW(policy.critic.parameters(),
                        lr=cfg.lr_critic, weight_decay=cfg.weight_decay)

    if resume_ckpt is not None:
        resume_path = Path(resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f'resume ckpt not found: {resume_path}')
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        policy.delta_head.load_state_dict(blob['delta_head_state'])
        policy.critic.load_state_dict(blob['critic_state'])
        if 'actor_opt_state' in blob:
            actor_opt.load_state_dict(blob['actor_opt_state'])
        if 'critic_opt_state' in blob:
            critic_opt.load_state_dict(blob['critic_opt_state'])
        print(f"[multi-ppo] resumed from {resume_path}", flush=True)
    elif cfg.init_radar_head_from:
        # Cross-run warm start (e.g. Phase 2 seeded from Phase 1
        # best.pt). Loads only weights — fresh optimizer for the new task.
        init_path = Path(cfg.init_radar_head_from)
        if not init_path.exists():
            raise FileNotFoundError(
                f'init_radar_head_from ckpt not found: {init_path}')
        policy.init_radar_head_from(init_path, device=device)
        print(f"[multi-ppo] radar head warm-started from {init_path} "
              f"(fresh optimizer state)", flush=True)

    cfg_dict = {
        'airport_name': cfg.airport_name,
        'runway': cfg.runway,
        'warmup_wpts': cfg.warmup_wpts,
        'max_timesteps_star_1_2': cfg.max_timesteps_star_1_2,
        'max_timesteps_star_3': cfg.max_timesteps_star_3,
        'success_reward': cfg.success_reward,
        'failure_reward': cfg.failure_reward,
        'gs_capture_buffer_ft': cfg.gs_capture_buffer_ft,
        'density_cutoff_nm': cfg.density_cutoff_nm,
        'density_n_bins': cfg.density_n_bins,
        'delta_hidden': cfg.delta_hidden,
        'value_hidden': cfg.value_hidden,
        'value_dropout': cfg.value_dropout,
        # Loop detector
        'loop_penalty_per_step': cfg.loop_penalty_per_step,
        'loop_prox_radius_nm': cfg.loop_prox_radius_nm,
        'loop_min_gap_steps': cfg.loop_min_gap_steps,
        'loop_min_detour_nm': cfg.loop_min_detour_nm,
        # Reward overrides forwarded to workers (same pattern as rl_ppo).
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
        'reward_per_star_recent_sr': per_star_recent_sr,
        # Phase-2 multi-plane bits (ignored by single-plane workers).
        'spawn_rate': cfg.spawn_rate,
        'collision_warning_penalty': cfg.collision_warning_penalty,
        'crash_extra_penalty': cfg.crash_extra_penalty,
        'replay_dir': str(cfg.run_dir / 'replays') if cfg.multi_plane else None,
    }

    import os
    n_workers = cfg.resolve_n_workers()
    t_pool_start = time.time()
    print(f"[multi-ppo] spawning {n_workers} workers "
          f"(mode={'MULTI' if cfg.multi_plane else 'SINGLE'}, "
          f"cpu_count={os.cpu_count() or 0})...", flush=True)
    if cfg.multi_plane:
        from rl_multiple.rollout import make_multi_pool, collect_multi_rollouts
        pool = make_multi_pool(cfg.ppo_ckpt, bc_seed, cfg_dict, n_workers)
    else:
        pool = make_rollout_pool(cfg.ppo_ckpt, bc_seed, cfg_dict, n_workers)
    print(f"[multi-ppo] pool ready in {time.time() - t_pool_start:.1f}s",
          flush=True)

    seed_offset = cfg.seed * 1_000_000
    best_success_rate = -1.0

    try:
        for it_local in range(cfg.n_iters):
            it = start_iter + it_local
            t_iter = time.time()

            policy_state = policy.state_dict_split()
            multi_stats = None
            if cfg.multi_plane:
                trajectories, multi_stats = collect_multi_rollouts(
                    pool=pool,
                    n_target_total=cfg.n_rollouts_per_iter,
                    policy_state=policy_state,
                    drop_truncated=cfg.drop_truncated,
                    verbose=(it % cfg.log_every == 0),
                )
            else:
                trajectories = collect_rollouts(
                    pool=pool, n_rollouts=cfg.n_rollouts_per_iter,
                    stars=cfg.stars, seed_offset=seed_offset,
                    policy_state=policy_state,
                    verbose=(it % cfg.log_every == 0),
                )
                seed_offset += cfg.n_rollouts_per_iter * 7
            if not trajectories:
                print(f"[multi-ppo] iter {it}: no trajectories — aborting",
                      flush=True)
                break

            batch = flatten_trajectories(trajectories,
                                          gamma=cfg.gamma, lam=cfg.gae_lambda)
            update_metrics = ppo_update(policy, actor_opt, critic_opt,
                                         batch, cfg, device)

            ep_summary = summarize_trajectories(trajectories)
            overall = ep_summary['overall']
            ret = batch['returns'].numpy()
            val = batch['values_old'].numpy()
            adv = batch['advantages'].numpy()
            explained_var = float(1.0 - (ret - val).var() / (ret.var() + 1e-8))
            batch_stats = {
                'returns/mean': float(ret.mean()),
                'returns/std': float(ret.std()),
                'advantages/mean': float(adv.mean()),
                'advantages/std': float(adv.std()),
                'value/explained_variance': explained_var,
            }
            health = delta_head_health(policy, batch['obs'].to(device))

            sec = time.time() - t_iter
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

            log_record = {**flat, 'per_star_nested': ep_summary}
            log_file.write(json.dumps(log_record) + '\n')
            log_file.flush()
            if metric_hook is not None:
                metric_hook(flat)

            if it_local % cfg.log_every == 0:
                # Phase-2 specific signals — present as extras when
                # multi_plane is on (and the values are populated).
                p2 = ''
                if cfg.multi_plane:
                    warn = overall.get('warning_seconds_total', 0)
                    mean_warn = overall.get('mean_warning_seconds_per_traj', 0.0)
                    n_crash = overall.get('n_crashes', 0)
                    crash_pct = overall.get('crash_rate', 0.0) * 100
                    pf_pct = overall.get('policy_failure_rate', 0.0) * 100
                    p2 = (f"  warn={warn}s ({mean_warn:.1f}/traj)  "
                          f"crashes={n_crash}({crash_pct:.0f}%)  "
                          f"pol_fail={pf_pct:.1f}%")
                print(f"[multi-ppo] it {it:>4d}  "
                      f"sr={overall['success_rate']:.3f}  "
                      f"steps/ep={overall['mean_steps']:.0f}  "
                      f"pi_loss={update_metrics['policy_loss']:+.4f}  "
                      f"v_loss={update_metrics['value_loss']:.4f}  "
                      f"kl={update_metrics['approx_kl']:.4f}  "
                      f"ev={explained_var:+.3f}  "
                      f"|μΔh|={abs(health['delta/mu_hdg_mean']):.2f}  "
                      f"σh={health['delta/sigma_hdg_mean']:.2f}"
                      + p2 +
                      f"  ({sec:.1f}s)", flush=True)

            if (it + 1) % cfg.save_every == 0 or it_local == cfg.n_iters - 1:
                ckpt_path = cfg.run_dir / f'iter_{it + 1:04d}.pt'
                torch.save({
                    'iter': it + 1,
                    'config': {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in asdict(cfg).items()},
                    'delta_head_state': policy.delta_head.state_dict(),
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
                        'delta_head_state': policy.delta_head.state_dict(),
                        'critic_state': policy.critic.state_dict(),
                        'overall_success_rate': overall['success_rate'],
                    }, cfg.run_dir / 'best.pt')

            # Every eval_six_pack_every iters, write the per-iter
            # rollouts as a 6-pack: rollouts.csv + trajectories.npz +
            # eval_metrics.json + summary.json + raw.json. PNGs render
            # locally after pull-back (matplotlib not in the Modal
            # training image). Uses this iter's training rollouts —
            # n_rollouts_per_iter ≈ 512 → ~85 per STAR, plenty for SR
            # and macro_green estimation.
            if (it + 1) % cfg.eval_six_pack_every == 0 \
                    or it_local == cfg.n_iters - 1:
                eval_dir = cfg.run_dir / f'iter_{it + 1:04d}_eval'
                from rl_multiple.eval_io import write_eval_six_pack
                m = write_eval_six_pack(trajectories, eval_dir, score=True)
                if m is not None:
                    o = m['overall']
                    print(f"  [6-pack] iter {it+1}  "
                          f"SR={o['success_rate']*100:.1f}%  "
                          f"macro_green={o['macro_pct_steps_in_green']*100:.1f}%"
                          f"  → {eval_dir}", flush=True)
    finally:
        pool.close()
        pool.join()
        log_file.close()
        if wandb_run is not None:
            import wandb
            wandb.finish()

    print(f"[multi-ppo] done. best_success_rate={best_success_rate:.3f}",
          flush=True)
    return {'best_success_rate': best_success_rate}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Multi-plane PPO — trains DeltaHead on top of frozen GMM')
    p.add_argument('--ppo-ckpt', type=str, default=None,
                   help='Frozen-GMM seed (PPO ckpt). Default: continuous_03 best.')
    p.add_argument('--bc-seed', type=str, default=None)
    p.add_argument('--run-dir', type=str, default=None)
    p.add_argument('--n-iters', type=int, default=None)
    p.add_argument('--n-rollouts', type=int, default=None)
    p.add_argument('--n-workers', type=int, default=None)
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
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-run', type=str, default=None)
    p.add_argument('--wandb-project', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--start-iter', type=int, default=0)
    # Reward knobs (subset that's useful per-block; mirror rl_ppo.train).
    p.add_argument('--everywhere-pen', type=float, default=None)
    p.add_argument('--step-pen-cap', type=float, default=None)
    p.add_argument('--step-pen-per-nm', type=float, default=None)
    p.add_argument('--out-of-zone-terminate', action='store_true')
    p.add_argument('--out-of-zone-max-consecutive', type=int, default=None)
    p.add_argument('--per-star-sr-scale', type=float, default=None)
    # Delta head clamps (useful for ablations later).
    p.add_argument('--delta-hdg-clamp-deg', type=float, default=None)
    p.add_argument('--delta-alt-clamp-kft', type=float, default=None)
    p.add_argument('--delta-spd-clamp-kt', type=float, default=None)
    p.add_argument('--eval-six-pack-every', type=int, default=None,
                   help='Write a 6-pack to run_dir/iter_NNNN_eval every '
                        'N iters (default 20).')
    # Phase-2 multi-plane flags
    p.add_argument('--multi-plane', action='store_true',
                   help='Enable Phase 2 (continuous multi-plane sim, '
                        'success=LANDED, collision-warning penalty active).')
    p.add_argument('--spawn-rate', type=int, default=None,
                   help='Phase 2 spawn cadence (seconds). Default 120.')
    p.add_argument('--collision-penalty', type=float, default=None,
                   help='Per-tick per-plane penalty while collision_warning '
                        'is True. Default 0.10.')
    p.add_argument('--init-radar-head-from', type=str, default=None,
                   help='Warm-start the radar head (delta head + critic) '
                        'from this ckpt (fresh optimizer). E.g. '
                        'rl_multiple/runs/phase1_v1/best.pt')
    return p


def main():
    args = _build_argparser().parse_args()
    cfg = MultiPPOConfig()
    overrides = {
        'ppo_ckpt': Path(args.ppo_ckpt) if args.ppo_ckpt else None,
        'bc_seed_ckpt': Path(args.bc_seed) if args.bc_seed else None,
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
        'out_of_zone_terminate': True if args.out_of_zone_terminate else None,
        'out_of_zone_max_consecutive': args.out_of_zone_max_consecutive,
        'per_star_sr_scale': args.per_star_sr_scale,
        'delta_hdg_clamp_deg': args.delta_hdg_clamp_deg,
        'delta_alt_clamp_kft': args.delta_alt_clamp_kft,
        'delta_spd_clamp_kt': args.delta_spd_clamp_kt,
        'eval_six_pack_every': args.eval_six_pack_every,
        'multi_plane': True if args.multi_plane else None,
        'spawn_rate': args.spawn_rate,
        'collision_warning_penalty': args.collision_penalty,
        'init_radar_head_from': args.init_radar_head_from,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.wandb:
        cfg.wandb = True
    summary = train(cfg,
                    resume_ckpt=args.resume,
                    start_iter=int(args.start_iter))
    print('summary:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
