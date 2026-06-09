"""bc_fm-specific training loop (joint 4-D CFM, position-only inputs).

The top-level `rl_bc/train.py` dispatcher calls `train_one_run(cfg, hook)`
from here when `cfg.family == 'bc_fm'`.
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from rl_bc.config import Config
from rl_bc.data import (BCDataset, Standardizer, describe_data, final_split,
                         load_cache)
from rl_bc.bc_fm.model import BCActor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)


def _build_model(cfg: Config) -> BCActor:
    return BCActor(
        input_indices=cfg.input_indices, hidden=cfg.hidden,
        dropout=cfg.dropout,
        fm_t_embed_dim=cfg.fm_t_embed_dim,
        fm_noise_scale=cfg.fm_noise_scale,
    )


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model: BCActor, loader: DataLoader, device: torch.device,
             head_weights: dict[str, float], cfg: Config) -> dict[str, float]:
    """Joint-FM eval. Returns val_loss + per-head MAE in physical units.

    The CFM velocity-field loss uses the same x_0 scale as training.
    `filter_loc_for_hdg=True` masks the two hdg dims of loss + MAE on
    loc==1 rows, mirroring what the runtime actually acts on.
    """
    model.eval()
    total_loss = 0.0
    n_rows = 0
    hdg_t, hdg_p, hdg_keep = [], [], []
    alt_t, alt_p = [], []
    spd_t, spd_p = [], []
    eval_rng = torch.Generator(device=device).manual_seed(0)

    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        t_hdg = batch['target_hdg_sincos'].to(device, non_blocking=True)
        t_alt = batch['target_alt_kft'].to(device, non_blocking=True)
        t_spd = batch['target_spd_norm'].to(device, non_blocking=True)
        keep_hdg = ((1.0 - batch['loc'].to(device, non_blocking=True))
                    if cfg.filter_loc_for_hdg else None)
        B = x.shape[0]

        # Joint 4-D target → standardized.
        t_joint = torch.stack([t_hdg[:, 0], t_hdg[:, 1], t_alt, t_spd], dim=-1)
        t_joint_std = model.standardize_target(t_joint)

        # CFM loss with the same noise scale as training.
        c = model.encode(x)
        x_0 = torch.randn_like(t_joint_std) * float(model.fm_noise_scale)
        t_uni = torch.rand(B, device=device, dtype=t_joint_std.dtype)
        x_t = (1.0 - t_uni).unsqueeze(-1) * x_0 + t_uni.unsqueeze(-1) * t_joint_std
        v_pred = model.velocity_field(x_t, t_uni, c)
        u_target = t_joint_std - x_0
        per_dim_sq = (v_pred - u_target) ** 2

        if keep_hdg is not None:
            dim_keep = torch.ones_like(per_dim_sq)
            dim_keep[:, 0] = keep_hdg
            dim_keep[:, 1] = keep_hdg
            per_dim_sq = per_dim_sq * dim_keep
            denom = torch.tensor([
                keep_hdg.sum().clamp(min=1.0), keep_hdg.sum().clamp(min=1.0),
                float(B), float(B),
            ], device=device, dtype=t_joint_std.dtype)
            per_dim_mean = per_dim_sq.sum(dim=0) / denom
        else:
            per_dim_mean = per_dim_sq.mean(dim=0)

        l_hdg = (per_dim_mean[0] + per_dim_mean[1]) * 0.5
        l_alt = per_dim_mean[2]
        l_spd = per_dim_mean[3]
        loss = (head_weights['hdg'] * l_hdg
                + head_weights['alt'] * l_alt
                + head_weights['spd'] * l_spd)
        total_loss += loss.item() * B
        n_rows += B

        # Sampled MAE in physical units (deterministic via fixed eval seed).
        sampled_std = model.sample(c, n_steps=cfg.fm_n_steps, generator=eval_rng)
        sampled = model.unstandardize_sample(sampled_std)
        hdg_p.append(sampled[:, :2].cpu().numpy())
        alt_p.append(sampled[:, 2].cpu().numpy())
        spd_p.append(sampled[:, 3].cpu().numpy())
        hdg_t.append(t_hdg.cpu().numpy())
        if keep_hdg is not None:
            hdg_keep.append(keep_hdg.cpu().numpy())
        alt_t.append(t_alt.cpu().numpy())
        spd_t.append(t_spd.cpu().numpy())

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
        'val_loss': total_loss / max(1, n_rows),
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
            "Run `python -m rl_bc.train --rebuild-cache --cache-only`.")

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cached = load_cache(cfg.cache_path)
    if cfg.final:
        train_idx, val_idx = final_split(cached, cfg.final_val_fraction, seed=cfg.seed)
    else:
        from rl_bc.data import kfold_episode_splits
        splits = kfold_episode_splits(cached, n_folds=cfg.n_folds, seed=cfg.seed)
        train_idx, val_idx = splits[cfg.fold]

    print(describe_data(cached, train_idx, val_idx, tag=cfg.run_dir.name),
          flush=True)

    standardizer = Standardizer.fit(cached.features, train_idx)
    train_ds = BCDataset(cached, train_idx, standardizer)
    val_ds = BCDataset(cached, val_idx, standardizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    model = _build_model(cfg).to(device)

    # Fit the 4-D target standardizer on the training subset and stash in buffers.
    t_hsin = cached.target_hdg_sin[train_idx]
    t_hcos = cached.target_hdg_cos[train_idx]
    t_akft = cached.target_alt_kft[train_idx]
    t_snrm = cached.target_spd_norm[train_idx]
    target_stack = np.stack([t_hsin, t_hcos, t_akft, t_snrm], axis=1).astype(np.float32)
    model.set_target_stats(target_stack.mean(axis=0), target_stack.std(axis=0))
    print(f"target_mean = {model.target_mean.tolist()}")
    print(f"target_std  = {model.target_std.tolist()}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    head_weights = {'hdg': cfg.hdg_loss_weight,
                    'alt': cfg.alt_loss_weight,
                    'spd': cfg.spd_loss_weight}
    steps_per_epoch = max(1, len(train_loader))
    scheduler = _cosine_with_warmup(
        optimizer,
        warmup_steps=cfg.warmup_epochs * steps_per_epoch,
        total_steps=cfg.epochs * steps_per_epoch,
    )

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.run_dir / 'best.pt'

    tag = cfg.run_dir.name
    print(f"[{tag}] arch = bc_fm joint-CFM  input{list(cfg.input_indices)} "
          f"h={cfg.hidden} (FM t_embed={cfg.fm_t_embed_dim}, n_steps={cfg.fm_n_steps}, "
          f"noise={cfg.fm_noise_scale})")
    print(f"[{tag}] train rows={len(train_ds)}  val rows={len(val_ds)}  device={device}")
    print(f"[{tag}] loss: joint 4-D CFM, per-dim weights "
          f"(hdg×{head_weights['hdg']:.2f}/2, alt×{head_weights['alt']:.2f}, "
          f"spd×{head_weights['spd']:.2f}); filter_loc_for_hdg={cfg.filter_loc_for_hdg}")

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
            t_hdg = batch['target_hdg_sincos'].to(device, non_blocking=True)
            t_alt = batch['target_alt_kft'].to(device, non_blocking=True)
            t_spd = batch['target_spd_norm'].to(device, non_blocking=True)
            keep_hdg = ((1.0 - batch['loc'].to(device, non_blocking=True))
                        if cfg.filter_loc_for_hdg else None)
            B = x.shape[0]

            t_joint = torch.stack([t_hdg[:, 0], t_hdg[:, 1], t_alt, t_spd], dim=-1)
            t_joint_std = model.standardize_target(t_joint)
            c = model.encode(x)
            x_0 = torch.randn_like(t_joint_std) * cfg.fm_noise_scale
            t_uniform = torch.rand(B, device=device, dtype=t_joint_std.dtype)
            x_t = (1.0 - t_uniform).unsqueeze(-1) * x_0 \
                + t_uniform.unsqueeze(-1) * t_joint_std
            u_target = t_joint_std - x_0
            v_pred = model.velocity_field(x_t, t_uniform, c)
            per_dim_sq = (v_pred - u_target) ** 2

            w = torch.tensor([head_weights['hdg'], head_weights['hdg'],
                              head_weights['alt'], head_weights['spd']],
                             device=device, dtype=t_joint_std.dtype)
            if keep_hdg is not None:
                dim_keep = torch.ones_like(per_dim_sq)
                dim_keep[:, 0] = keep_hdg
                dim_keep[:, 1] = keep_hdg
                per_dim_sq = per_dim_sq * dim_keep
                denom = torch.tensor([
                    keep_hdg.sum().clamp(min=1.0), keep_hdg.sum().clamp(min=1.0),
                    float(B), float(B),
                ], device=device, dtype=t_joint_std.dtype)
                per_dim_mean = per_dim_sq.sum(dim=0) / denom
            else:
                per_dim_mean = per_dim_sq.mean(dim=0)
            loss = (w * per_dim_mean).sum()
            l_hdg = (per_dim_mean[0] + per_dim_mean[1]) * 0.5
            l_alt = per_dim_mean[2]
            l_spd = per_dim_mean[3]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * B
            running_n += B
            global_step += 1

            if metric_hook is not None and (global_step % cfg.log_every_n_steps == 0):
                metric_hook({
                    'train/loss': loss.item(),
                    'train/loss_hdg': l_hdg.item(),
                    'train/loss_alt': l_alt.item(),
                    'train/loss_spd': l_spd.item(),
                    'train/grad_norm': float(grad_norm),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'step': global_step, 'epoch': epoch,
                })

        train_loss = running_loss / max(1, running_n)
        epoch_secs = time.time() - epoch_start

        val_metrics = evaluate(model, val_loader, device, head_weights, cfg)
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
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
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
                'head_weights': head_weights,
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


