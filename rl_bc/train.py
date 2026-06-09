"""Local training dispatcher.

Loads the requested config (which carries `FAMILY`), then hands off to the
family's own training module (`rl_bc.<family>.train.train_one_run`). New
families are added by dropping a new package + config under `rl_bc/`.

Run:

    python -m rl_bc.train --config bc_fm_single --final
    python -m rl_bc.train --config bc_gmm_single --final
    python -m rl_bc.train --config bc_fm_single --rebuild-cache --cache-only

For cloud / W&B-logged runs:

    modal run rl_bc/modal_train.py --config bc_fm_single
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from rl_bc.config import HUMAN_DATA_DIR, REPO_ROOT, Config
from rl_bc.configs import load_config
from rl_bc.data import build_cache, cache_is_fresh


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='BC training dispatcher (bc_fm, bc_gmm, …)')
    p.add_argument('--config', type=str, required=True,
                   help='Config name in rl_bc/configs/ (sans .py).')
    p.add_argument('--final', action='store_true',
                   help='Train on (almost) all data; ignored if --fold given.')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--cache-only', action='store_true')
    p.add_argument('--human-data-dir', type=str, default=str(HUMAN_DATA_DIR))
    p.add_argument('--cache-path', type=str, default=None)
    p.add_argument('--run-dir', type=str, default=None,
                   help='Override the auto-derived rl_bc/runs/<NAME>/ path.')
    p.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'])
    return p


def _cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    exp = load_config(args.config)
    cfg.name = exp.NAME
    cfg.family = exp.FAMILY
    cfg.epochs = exp.EPOCHS
    cfg.batch_size = exp.BATCH_SIZE
    cfg.lr = exp.LR
    cfg.weight_decay = exp.WEIGHT_DECAY
    cfg.warmup_epochs = exp.WARMUP_EPOCHS
    cfg.grad_clip = exp.GRAD_CLIP
    cfg.early_stop_patience = exp.EARLY_STOP_PATIENCE
    cfg.seed = exp.SEED
    cfg.final_val_fraction = exp.FINAL_VAL_FRACTION
    cfg.dropout = exp.DROPOUT
    cfg.input_indices = tuple(exp.INPUT_INDICES)
    cfg.hidden = exp.HIDDEN
    cfg.hdg_loss_weight = exp.HDG_LOSS_WEIGHT
    cfg.alt_loss_weight = exp.ALT_LOSS_WEIGHT
    cfg.spd_loss_weight = exp.SPD_LOSS_WEIGHT
    cfg.fm_t_embed_dim = int(getattr(exp, 'FM_T_EMBED_DIM', 16))
    cfg.fm_n_steps = int(getattr(exp, 'FM_N_STEPS', 10))
    cfg.fm_noise_scale = float(getattr(exp, 'FM_NOISE_SCALE', 1.0))
    cfg.n_components = int(getattr(exp, 'N_COMPONENTS', 4))
    cfg.mixture_balance_weight = float(getattr(exp, 'MIXTURE_BALANCE_WEIGHT', 0.0))
    cfg.row_entropy_weight = float(getattr(exp, 'ROW_ENTROPY_WEIGHT', 0.0))
    cfg.row_entropy_target = float(getattr(exp, 'ROW_ENTROPY_TARGET', 0.0))
    cfg.focal_gamma = float(getattr(exp, 'FOCAL_GAMMA', 0.0))
    cfg.focal_threshold = float(getattr(exp, 'FOCAL_THRESHOLD', 0.0))
    cfg.focal_temperature = float(getattr(exp, 'FOCAL_TEMPERATURE', 1.0))
    cfg.heading_dropout_prob = float(getattr(exp, 'HEADING_DROPOUT_PROB', 0.0))
    cfg.alt_dropout_prob = float(getattr(exp, 'ALT_DROPOUT_PROB', 0.0))
    cfg.spd_dropout_prob = float(getattr(exp, 'SPD_DROPOUT_PROB', 0.0))
    cfg.dataset_repeat = int(getattr(exp, 'DATASET_REPEAT', 1))
    cfg.input_noise_heading_deg = float(getattr(exp, 'INPUT_NOISE_HEADING_DEG', 0.0))
    cfg.input_noise_alt_ft = float(getattr(exp, 'INPUT_NOISE_ALT_FT', 0.0))
    cfg.input_noise_spd_kt = float(getattr(exp, 'INPUT_NOISE_SPD_KT', 0.0))
    cfg.noise_clamp_alt_min_ft = float(getattr(exp, 'NOISE_CLAMP_ALT_MIN_FT', cfg.noise_clamp_alt_min_ft))
    cfg.noise_clamp_alt_max_ft = float(getattr(exp, 'NOISE_CLAMP_ALT_MAX_FT', cfg.noise_clamp_alt_max_ft))
    cfg.noise_clamp_spd_min_kt = float(getattr(exp, 'NOISE_CLAMP_SPD_MIN_KT', cfg.noise_clamp_spd_min_kt))
    cfg.noise_clamp_spd_max_kt = float(getattr(exp, 'NOISE_CLAMP_SPD_MAX_KT', cfg.noise_clamp_spd_max_kt))
    cfg.filter_loc_for_hdg = bool(getattr(exp, 'FILTER_LOC_FOR_HDG', False))
    cfg.run_dir = REPO_ROOT / 'rl_bc' / 'runs' / exp.NAME

    cfg.fold = args.fold
    cfg.final = args.final
    cfg.n_folds = args.n_folds
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.lr is not None: cfg.lr = args.lr
    if args.seed is not None: cfg.seed = args.seed
    if args.cache_path: cfg.cache_path = Path(args.cache_path)
    if args.run_dir: cfg.run_dir = Path(args.run_dir)
    return cfg


def train_one_fold(cfg: Config, device: torch.device | None = None,
                   metric_hook=None) -> dict:
    """Dispatch to the family's `train_one_run`. The Modal `train_remote`
    wrapper calls this entry point too, so family changes don't need any
    plumbing changes upstream.
    """
    if not cfg.family:
        raise ValueError("cfg.family is empty; configs must set FAMILY")
    module = importlib.import_module(f"rl_bc.{cfg.family}.train")
    return module.train_one_run(cfg, device=device, metric_hook=metric_hook)


def main():
    args = _build_argparser().parse_args()
    cfg = _cfg_from_args(args)

    human_dir = Path(args.human_data_dir)
    should_rebuild = (
        args.rebuild_cache or args.cache_only
        or not cache_is_fresh(cfg.cache_path, human_dir)
    )
    if should_rebuild:
        if not (args.rebuild_cache or args.cache_only) and cfg.cache_path.exists():
            print(f"cache stale (CSVs in {human_dir} changed) — auto-rebuilding")
        elif not cfg.cache_path.exists():
            print(f"cache missing at {cfg.cache_path} — auto-rebuilding")
        build_cache(cfg, human_dir=human_dir, cache_path=cfg.cache_path)
    else:
        print(f"cache fresh — reusing {cfg.cache_path}")
    if args.cache_only:
        return

    device = torch.device(args.device) if args.device else None
    summary = train_one_fold(cfg, device=device)
    print('summary:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
