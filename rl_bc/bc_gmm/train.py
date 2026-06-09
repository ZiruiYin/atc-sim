"""bc_gmm-specific training loop.

NLL on the joint Gaussian-mixture density. The LOC mask sets `keep_dims`
per row so loc==1 rows contribute only the marginal log-likelihood over
the alt and spd dimensions (the autopilot-locked hdg target on those
rows isn't a real human command).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from rl_bc.config import Config
from rl_bc.data import (BCDataset, Standardizer, describe_data, final_split,
                         load_cache)
from rl_bc.bc_gmm.model import BCActor


class _RepeatedDataset(torch.utils.data.Dataset):
    """Cheap multiplicative wrapper: reports `n * len(base)` rows and
    delegates each access modulo the base length. Lets stochastic
    augmentations (noise, dropout) produce N effectively-independent
    noisy samples per row per epoch without touching the underlying
    BCDataset / cache."""

    def __init__(self, base, n: int):
        self.base = base
        self.n = int(n)
        self._L = len(base)

    def __len__(self) -> int:
        return self._L * self.n

    def __getitem__(self, i):
        return self.base[i % self._L]


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)


def _build_model(cfg: Config) -> BCActor:
    return BCActor(
        input_indices=cfg.input_indices,
        hidden=cfg.hidden,
        n_components=cfg.n_components,
        dropout=cfg.dropout,
    )


def _stack_target(batch, device) -> torch.Tensor:
    """(B, 4) joint target — same dim order as bc_fm."""
    t_hdg = batch['target_hdg_sincos'].to(device, non_blocking=True)
    t_alt = batch['target_alt_kft'].to(device, non_blocking=True)
    t_spd = batch['target_spd_norm'].to(device, non_blocking=True)
    return torch.stack([t_hdg[:, 0], t_hdg[:, 1], t_alt, t_spd], dim=-1)


def _per_row_dim_mask(batch, device, action_dim: int,
                      filter_loc_for_hdg: bool) -> torch.Tensor:
    """(B, D) 0/1 mask. When the LOC filter is on, hdg dims (0, 1) are zeroed
    for rows where loc==1; alt and spd dims always pass.
    """
    B = batch['x'].shape[0]
    keep = torch.ones(B, action_dim, device=device, dtype=torch.float32)
    if filter_loc_for_hdg:
        loc = batch['loc'].to(device, non_blocking=True)         # (B,)
        # broadcast: keep[:, 0:2] *= (1 - loc)
        mask = (1.0 - loc).unsqueeze(-1)
        keep[:, :2] = keep[:, :2] * mask
    return keep


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model: BCActor, loader: DataLoader, device: torch.device,
             cfg: Config) -> dict[str, float]:
    """Returns val_loss (mean NLL with per-dim weighting + LOC mask) plus MAE
    in physical units. MAE is computed from per-batch sampled actions with a
    fixed eval seed so the metric is reproducible.
    """
    model.eval()
    total_nll = 0.0
    n_rows = 0
    hdg_t, hdg_p, hdg_keep = [], [], []
    alt_t, alt_p = [], []
    spd_t, spd_p = [], []
    eval_rng = torch.Generator(device=device).manual_seed(0)
    D = model.ACTION_DIM

    weight_per_dim = torch.tensor(
        [cfg.hdg_loss_weight, cfg.hdg_loss_weight,
         cfg.alt_loss_weight, cfg.spd_loss_weight],
        device=device, dtype=torch.float32,
    )

    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        B = x.shape[0]
        t_joint = _stack_target(batch, device)
        t_joint_std = model.standardize_target(t_joint)
        keep = _per_row_dim_mask(batch, device, D, cfg.filter_loc_for_hdg)

        c = model.encode(x)
        # Weighted NLL: weight is applied per-dim before summing inside log_prob.
        # Easiest equivalent: pre-weight `keep` by weight_per_dim.
        weighted_keep = keep * weight_per_dim
        log_p = model.log_prob(t_joint_std, c, keep_dims=weighted_keep)   # (B,)
        nll = -log_p.mean()

        total_nll += nll.item() * B
        n_rows += B

        sampled_std = model.sample(c, generator=eval_rng, deterministic=False)
        sampled = model.unstandardize_sample(sampled_std)
        hdg_p.append(sampled[:, :2].cpu().numpy())
        alt_p.append(sampled[:, 2].cpu().numpy())
        spd_p.append(sampled[:, 3].cpu().numpy())
        hdg_t.append(batch['target_hdg_sincos'].numpy())
        if cfg.filter_loc_for_hdg:
            hdg_keep.append((1.0 - batch['loc']).numpy())
        alt_t.append(batch['target_alt_kft'].numpy())
        spd_t.append(batch['target_spd_norm'].numpy())

    hdg_t = np.concatenate(hdg_t); hdg_p = np.concatenate(hdg_p)
    alt_t = np.concatenate(alt_t); alt_p = np.concatenate(alt_p)
    spd_t = np.concatenate(spd_t); spd_p = np.concatenate(spd_p)

    pred_hdg_deg = (np.degrees(np.arctan2(hdg_p[:, 0], hdg_p[:, 1])) + 360.0) % 360.0
    true_hdg_deg = (np.degrees(np.arctan2(hdg_t[:, 0], hdg_t[:, 1])) + 360.0) % 360.0
    diff = (pred_hdg_deg - true_hdg_deg + 540.0) % 360.0 - 180.0
    if hdg_keep:
        keep_arr = np.concatenate(hdg_keep) > 0.5
        hdg_mae_deg = float(np.mean(np.abs(diff[keep_arr]))) if keep_arr.any() else 0.0
    else:
        hdg_mae_deg = float(np.mean(np.abs(diff)))

    return {
        'val_loss': total_nll / max(1, n_rows),
        'val_hdg_mae_deg': hdg_mae_deg,
        'val_alt_mae_ft': float(np.mean(np.abs(alt_p - alt_t))) * 1000.0,
        'val_spd_mae_kt': float(np.mean(np.abs(spd_p - spd_t))) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #


def train_one_run(cfg: Config, device: torch.device | None = None,
                  metric_hook: Callable[[dict], None] | None = None
                  ) -> dict[str, Any]:
    if not cfg.cache_path.exists():
        raise FileNotFoundError(
            f"cache not found at {cfg.cache_path}. "
            "Run `python -m rl_bc.train --config <name> --rebuild-cache --cache-only`.")

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cached = load_cache(cfg.cache_path)
    if cfg.final:
        train_idx, val_idx = final_split(cached, cfg.final_val_fraction, seed=cfg.seed)
    else:
        from rl_bc.data import kfold_episode_splits
        splits = kfold_episode_splits(cached, n_folds=cfg.n_folds, seed=cfg.seed)
        train_idx, val_idx = splits[cfg.fold]

    # Source ablation: drop rows from excluded sources from both splits so
    # the model trains/validates only on the kept sources. The episode
    # split is source-agnostic, so the kept val fraction is preserved.
    if cfg.train_sources:
        from rl_bc.data import SOURCE_LABEL
        allowed = np.asarray(cfg.train_sources, dtype=np.int64)
        src = cached.source
        n_tr0, n_va0 = len(train_idx), len(val_idx)
        train_idx = train_idx[np.isin(src[train_idx], allowed)]
        val_idx = val_idx[np.isin(src[val_idx], allowed)]
        kept = ', '.join(SOURCE_LABEL.get(int(s), str(s)) for s in allowed)
        print(f"[{cfg.run_dir.name}] source filter → keep [{kept}]  "
              f"train rows {n_tr0}→{len(train_idx)}  "
              f"val rows {n_va0}→{len(val_idx)}", flush=True)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(
                f"source filter {cfg.train_sources} left an empty split "
                f"(train={len(train_idx)}, val={len(val_idx)})")

    print(describe_data(cached, train_idx, val_idx, tag=cfg.run_dir.name),
          flush=True)

    standardizer = Standardizer.fit(cached.features, train_idx)
    train_ds = BCDataset(cached, train_idx, standardizer)
    val_ds = BCDataset(cached, val_idx, standardizer)

    if cfg.dataset_repeat > 1:
        train_iter_ds: torch.utils.data.Dataset = _RepeatedDataset(
            train_ds, cfg.dataset_repeat)
        print(f"[train] dataset_repeat={cfg.dataset_repeat} → "
              f"iter rows {len(train_ds)} × {cfg.dataset_repeat} = "
              f"{len(train_iter_ds)} (noise/dropout resampled each access)")
    else:
        train_iter_ds = train_ds

    train_loader = DataLoader(
        train_iter_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    model = _build_model(cfg).to(device)
    D = model.ACTION_DIM

    # Fit per-dim target standardizer on the training subset; stash into buffers.
    t_hsin = cached.target_hdg_sin[train_idx]
    t_hcos = cached.target_hdg_cos[train_idx]
    t_akft = cached.target_alt_kft[train_idx]
    t_snrm = cached.target_spd_norm[train_idx]
    target_stack = np.stack([t_hsin, t_hcos, t_akft, t_snrm], axis=1).astype(np.float32)
    model.set_target_stats(target_stack.mean(axis=0), target_stack.std(axis=0))
    print(f"target_mean = {model.target_mean.tolist()}")
    print(f"target_std  = {model.target_std.tolist()}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = _cosine_with_warmup(
        optimizer,
        warmup_steps=cfg.warmup_epochs * steps_per_epoch,
        total_steps=cfg.epochs * steps_per_epoch,
    )

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.run_dir / 'best.pt'

    weight_per_dim = torch.tensor(
        [cfg.hdg_loss_weight, cfg.hdg_loss_weight,
         cfg.alt_loss_weight, cfg.spd_loss_weight],
        device=device, dtype=torch.float32,
    )

    # Pre-compute noise sigmas in standardized units (alt/spd) and radians
    # (heading). standardizer.std at cols 4, 5 gives the spread of the
    # physical alt/spd values in standardized space → divide physical σ by it.
    _noise_hdg_rad = math.radians(cfg.input_noise_heading_deg)
    _noise_alt_std = ((cfg.input_noise_alt_ft / 1000.0)
                      / max(float(standardizer.std[4]), 1e-6)
                      if cfg.input_noise_alt_ft > 0 else 0.0)
    _noise_spd_std = ((cfg.input_noise_spd_kt / 100.0)
                      / max(float(standardizer.std[5]), 1e-6)
                      if cfg.input_noise_spd_kt > 0 else 0.0)

    # Physical → standardized clamp bounds. col 4 stores alt/1000 then
    # standardizes; col 5 stores (ias-200)/100 then standardizes. Reverse
    # both transforms to put the configured physical caps into std-space.
    _alt_clamp = (cfg.noise_clamp_alt_max_ft > cfg.noise_clamp_alt_min_ft
                  and _noise_alt_std > 0)
    _spd_clamp = (cfg.noise_clamp_spd_max_kt > cfg.noise_clamp_spd_min_kt
                  and _noise_spd_std > 0)
    if _alt_clamp:
        _alt_lo_std = ((cfg.noise_clamp_alt_min_ft / 1000.0) - float(standardizer.mean[4])) / float(standardizer.std[4])
        _alt_hi_std = ((cfg.noise_clamp_alt_max_ft / 1000.0) - float(standardizer.mean[4])) / float(standardizer.std[4])
    if _spd_clamp:
        _spd_lo_std = (((cfg.noise_clamp_spd_min_kt - 200.0) / 100.0) - float(standardizer.mean[5])) / float(standardizer.std[5])
        _spd_hi_std = (((cfg.noise_clamp_spd_max_kt - 200.0) / 100.0) - float(standardizer.mean[5])) / float(standardizer.std[5])

    tag = cfg.run_dir.name
    if _noise_hdg_rad > 0 or _noise_alt_std > 0 or _noise_spd_std > 0:
        print(f"[{tag}] input-noise aug: "
              f"hdg={cfg.input_noise_heading_deg:.1f}deg (={_noise_hdg_rad:.3f}rad)  "
              f"alt={cfg.input_noise_alt_ft:.0f}ft (={_noise_alt_std:.3f}std)  "
              f"spd={cfg.input_noise_spd_kt:.0f}kt (={_noise_spd_std:.3f}std)")
        if _alt_clamp:
            print(f"[{tag}] alt clamp: [{cfg.noise_clamp_alt_min_ft:.0f}, "
                  f"{cfg.noise_clamp_alt_max_ft:.0f}] ft  "
                  f"= [{_alt_lo_std:.3f}, {_alt_hi_std:.3f}] std")
        if _spd_clamp:
            print(f"[{tag}] spd clamp: [{cfg.noise_clamp_spd_min_kt:.0f}, "
                  f"{cfg.noise_clamp_spd_max_kt:.0f}] kt  "
                  f"= [{_spd_lo_std:.3f}, {_spd_hi_std:.3f}] std")
    print(f"[{tag}] arch = bc_gmm  K={cfg.n_components}  input{list(cfg.input_indices)} "
          f"h={cfg.hidden}  diagonal-covariance GMM with target standardization")
    print(f"[{tag}] train rows={len(train_ds)}  val rows={len(val_ds)}  device={device}")
    print(f"[{tag}] loss: NLL  per-dim weights "
          f"hdg×{cfg.hdg_loss_weight:.2f}/2, alt×{cfg.alt_loss_weight:.2f}, "
          f"spd×{cfg.spd_loss_weight:.2f};  filter_loc_for_hdg={cfg.filter_loc_for_hdg}")
    if cfg.mixture_balance_weight > 0:
        print(f"[{tag}] regularizer: + {cfg.mixture_balance_weight:.3f} × "
              f"(log K - H(π̄))  ← batch-mean load balancing")
    if cfg.row_entropy_weight > 0:
        print(f"[{tag}] regularizer: + {cfg.row_entropy_weight:.3f} × "
              f"ReLU({cfg.row_entropy_target:.2f} - H_row)² "
              f" ← per-row entropy floor")
    # σ floor and κ ceiling are now structural (softplus reparam in
    # model.gmm_params); no loss term needed.
    from rl_bc.bc_gmm.model import LOG_STD_FLOOR, LOG_KAPPA_CEILING
    print(f"[{tag}] structural bounds: "
          f"σ ≥ {math.exp(LOG_STD_FLOOR):.3f} std  "
          f"κ ≤ {math.exp(LOG_KAPPA_CEILING):.1f} "
          f"(ang_std ≥ {math.degrees(1.0/math.sqrt(math.exp(LOG_KAPPA_CEILING))):.1f}°)")

    best_metric = -math.inf
    best_epoch = -1
    epochs_since_improvement = 0
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_n = 0
        for batch in train_loader:
            x = batch['x'].to(device, non_blocking=True)
            B = x.shape[0]

            # Apply noise THEN per-channel dropout so dropout (which sets
            # the channel to its standardized mean = "missing" signal)
            # cleanly overrides the noise on any dropped channel.
            _aug = (_noise_hdg_rad > 0 or _noise_alt_std > 0 or _noise_spd_std > 0
                    or cfg.heading_dropout_prob > 0.0
                    or cfg.alt_dropout_prob > 0.0
                    or cfg.spd_dropout_prob > 0.0)
            if _aug:
                x = x.clone()
                # ---- Gaussian noise on current state channels ----
                if _noise_hdg_rad > 0:
                    theta = torch.atan2(x[:, 6], x[:, 7])
                    theta = theta + torch.randn_like(theta) * _noise_hdg_rad
                    x[:, 6] = torch.sin(theta)
                    x[:, 7] = torch.cos(theta)
                if _noise_alt_std > 0:
                    x[:, 4] = x[:, 4] + torch.randn(B, device=device) * _noise_alt_std
                if _noise_spd_std > 0:
                    x[:, 5] = x[:, 5] + torch.randn(B, device=device) * _noise_spd_std
                # ---- physical-envelope clamps (keep Gaussian tails inside ATC limits) ----
                if _alt_clamp:
                    x[:, 4] = x[:, 4].clamp_(_alt_lo_std, _alt_hi_std)
                if _spd_clamp:
                    x[:, 5] = x[:, 5].clamp_(_spd_lo_std, _spd_hi_std)
                # ---- per-channel dropout ----
                if cfg.heading_dropout_prob > 0.0:
                    m = torch.rand(B, device=device) < cfg.heading_dropout_prob
                    if m.any():
                        x[m, 6] = 0.0
                        x[m, 7] = 0.0
                if cfg.alt_dropout_prob > 0.0:
                    m = torch.rand(B, device=device) < cfg.alt_dropout_prob
                    if m.any():
                        x[m, 4] = 0.0
                if cfg.spd_dropout_prob > 0.0:
                    m = torch.rand(B, device=device) < cfg.spd_dropout_prob
                    if m.any():
                        x[m, 5] = 0.0

            t_joint = _stack_target(batch, device)
            t_joint_std = model.standardize_target(t_joint)
            keep = _per_row_dim_mask(batch, device, D, cfg.filter_loc_for_hdg)
            weighted_keep = keep * weight_per_dim

            c = model.encode(x)
            params = model.gmm_params(c)
            logits, *_ = params
            log_p = model.log_prob_from_params(t_joint_std, params,
                                               keep_dims=weighted_keep)
            nll_per_row = -log_p                              # (B,)

            # Focal NLL: down-weight rows the model already fits well
            # ("easy" = low NLL). Hardness = sigmoid((NLL - thr) / temp).
            # Standard generalization of focal loss to continuous targets;
            # fights the imbalance where ~85% of rows are steady-state.
            if cfg.focal_gamma > 0.0:
                if cfg.focal_threshold > 0.0:
                    thr = cfg.focal_threshold
                else:
                    # Adaptive: per-batch median. .detach() so the threshold
                    # doesn't backprop weird gradients through the weight.
                    thr = nll_per_row.detach().median()
                hardness = torch.sigmoid(
                    (nll_per_row.detach() - thr) / max(cfg.focal_temperature, 1e-6)
                )
                focal_weight = hardness.pow(cfg.focal_gamma)
                # Normalize so the loss scale is comparable across γ values
                # (otherwise a high γ shrinks loss magnitude → effective LR drops).
                wnorm = focal_weight.mean().clamp(min=1e-6)
                nll = (focal_weight * nll_per_row).sum() / (B * wnorm)
            else:
                nll = nll_per_row.mean()

            loss = nll

            # σ floor and κ ceiling are now STRUCTURAL bounds, baked into
            # gmm_params via softplus reparameterization — no penalty terms
            # needed (they're impossible to violate by construction).
            # Only mixture-distribution regularizers remain here.

            if cfg.mixture_balance_weight > 0.0:
                # log K - H(mean_pi); fights "everyone uses same comp" collapse
                pi = torch.softmax(logits, dim=-1)
                mean_pi = pi.mean(dim=0).clamp(min=1e-8)
                H_batch = -(mean_pi * mean_pi.log()).sum()
                bal = math.log(model.K) - H_batch
                loss = loss + cfg.mixture_balance_weight * bal
            else:
                bal = torch.tensor(0.0, device=device)

            if cfg.row_entropy_weight > 0.0:
                # ReLU(H_target - H_row).pow(2); fights per-state collapse
                pi_row = torch.softmax(logits, dim=-1).clamp(min=1e-8)
                H_row = -(pi_row * pi_row.log()).sum(dim=-1)               # (B,)
                row_pen = F.relu(cfg.row_entropy_target - H_row).pow(2).mean()
                loss = loss + cfg.row_entropy_weight * row_pen
            else:
                row_pen = torch.tensor(0.0, device=device)

            optimizer.zero_grad(set_to_none=True)
            # Skip non-finite losses (rare numerical edge under heavy aug).
            # Stepping the optimizer with a NaN/Inf loss corrupts every
            # weight; skip the batch instead so one bad sample doesn't
            # take the whole run down.
            if not torch.isfinite(loss):
                global_step += 1
                continue
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * B
            running_n += B
            global_step += 1

            if metric_hook is not None and (global_step % cfg.log_every_n_steps == 0):
                metric_hook({
                    'train/loss': loss.item(),
                    'train/nll': nll.item(),
                    'train/mixture_balance': bal.item(),
                    'train/row_entropy_penalty': row_pen.item(),
                    'train/grad_norm': float(grad_norm),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'step': global_step, 'epoch': epoch,
                })

        train_loss = running_loss / max(1, running_n)
        epoch_secs = time.time() - epoch_start

        val_metrics = evaluate(model, val_loader, device, cfg)
        hdg_score = max(0.0, 1.0 - val_metrics['val_hdg_mae_deg'] / 60.0)
        alt_score = max(0.0, 1.0 - val_metrics['val_alt_mae_ft'] / 10000.0)
        spd_score = max(0.0, 1.0 - val_metrics['val_spd_mae_kt'] / 100.0)
        primary = (hdg_score + alt_score + spd_score) / 3.0
        improved = primary > best_metric

        log = {'epoch': epoch, 'step': global_step, 'train_loss': train_loss,
               'epoch_secs': epoch_secs, **val_metrics,
               'val_hdg_score': hdg_score, 'val_score_mean': primary,
               'best_val_score_mean': max(best_metric, primary)}
        print(
            f"[{tag}] ep {epoch:02d}/{cfg.epochs} "
            f"train_nll={train_loss:.4f} "
            f"val_nll={val_metrics['val_loss']:.4f} "
            f"hdgMAE={val_metrics['val_hdg_mae_deg']:>5.2f}° "
            f"altMAE={val_metrics['val_alt_mae_ft']:>5.0f}ft "
            f"spdMAE={val_metrics['val_spd_mae_kt']:>5.1f}kt "
            f"{'*' if improved else ''} ({epoch_secs:.1f}s)"
        )
        if metric_hook is not None:
            metric_hook(log)

        if improved:
            best_metric = primary
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'standardizer_mean': standardizer.mean,
                'standardizer_std': standardizer.std,
                'config': asdict(cfg) | {
                    'cache_path': str(cfg.cache_path),
                    'run_dir': str(cfg.run_dir),
                },
                'val_metrics': val_metrics,
            }, ckpt_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                print(f"[{tag}] early stop at epoch {epoch} "
                      f"(no improvement for {epochs_since_improvement} epochs).")
                break

    summary = {
        'mode': 'final' if cfg.final else 'fold',
        'fold': None if cfg.final else cfg.fold,
        'best_epoch': best_epoch,
        'best_val_score_mean': best_metric,
        'ckpt_path': str(ckpt_path),
    }
    (cfg.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
