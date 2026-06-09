"""Shared Modal infrastructure for the BC trainer.

Defines the app, container image, persistent volume (`atc-bc`), W&B secret,
and the remote `train_remote` function. Imported by `rl_bc/modal_train.py`.

One-time setup
--------------

    pip install modal
    modal token new
    modal secret create wandb WANDB_API_KEY=<your_key>
    modal volume create atc-bc

After that, the local `modal_train.py` entrypoint takes care of everything
(syncs human_data, allocates a fresh run number, trains, logs to W&B, pulls
the best checkpoint back).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import modal


# Resolve the modal CLI via the active Python interpreter so it works in
# any shell (some shells don't have `modal` on PATH but `python -m modal`
# always resolves to the same modal version that's installed in env).
_MODAL_CMD = [sys.executable, "-m", "modal"]


APP_NAME = "atc-bc"
VOLUME_NAME = "atc-bc"
WANDB_SECRET_NAME = "wandb"
REMOTE_REPO_ROOT = "/root/atc-sim"
REMOTE_VOLUME_MOUNT = "/root/atc-sim/_modal"  # cache + checkpoints + human_data

_LOCAL_SYNC_MARKER = Path("rl_bc/.last_modal_data_sync")
_LOCAL_HUMAN_DATA = Path("human_data")
_PREPARED_REL = Path("prepared") / "combined.csv"   # produced by
                                                    # rl_bc.eval.prepare_training_data


# --------------------------------------------------------------------------- #
# Local-side helpers (run BEFORE / AFTER the remote train function)
# --------------------------------------------------------------------------- #


def sync_local_data_to_modal_volume(force: bool = False) -> bool:
    """Upload data to the Modal volume iff its fingerprint changed.

    Two modes:

    - If `human_data/prepared/combined.csv` exists (produced by the local
      `rl_bc.eval.prepare_training_data` step), upload ONLY that file —
      the cache builder treats it as the sole input. Fingerprint reflects
      just that one file; this is the path we want for new pipelines.

    - Otherwise fall back to syncing the entire `human_data/` tree (legacy
      behavior for runs that haven't switched to the prepared CSV).

    Returns True iff we actually uploaded (the caller should then set
    `rebuild_cache=True` on the remote run so the stale volume cache doesn't
    get reused).
    """
    from rl_bc.data import data_fingerprint

    if not _LOCAL_HUMAN_DATA.exists():
        print(f"  ! {_LOCAL_HUMAN_DATA} does not exist locally; skipping sync.")
        return False

    prepared = _LOCAL_HUMAN_DATA / _PREPARED_REL
    fp = data_fingerprint(_LOCAL_HUMAN_DATA)
    if not force and _LOCAL_SYNC_MARKER.exists() and _LOCAL_SYNC_MARKER.read_text().strip() == fp:
        scope = "prepared CSV" if prepared.exists() else "human_data/"
        print(f"  ✓ {scope} unchanged since last sync (fp={fp[:12]})")
        return False

    if prepared.exists():
        print(f"  → {prepared} changed (fp={fp[:12]}); uploading prepared CSV only...")
        subprocess.run(
            [*_MODAL_CMD, "volume", "put", VOLUME_NAME,
             str(prepared), f"/human_data/{_PREPARED_REL.as_posix()}", "--force"],
            check=True,
        )
    else:
        print(f"  → human_data/ changed (fp={fp[:12]}); uploading full tree to '{VOLUME_NAME}'...")
        subprocess.run(
            [*_MODAL_CMD, "volume", "put", VOLUME_NAME,
             str(_LOCAL_HUMAN_DATA), "/human_data", "--force"],
            check=True,
        )
    subprocess.run(
        [*_MODAL_CMD, "volume", "rm", VOLUME_NAME, "/cache/bc_dataset.npz"],
        check=False,
    )
    _LOCAL_SYNC_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _LOCAL_SYNC_MARKER.write_text(fp)
    print(f"  ✓ data synced; volume cache invalidated")
    return True


def next_run_number(config_name: str) -> int:
    """Query the Modal volume for the next free run_N under `runs/<config_name>/`.

    Returns 1 if no prior runs exist.
    """
    proc = subprocess.run(
        [*_MODAL_CMD, "volume", "ls", VOLUME_NAME, f"/runs/{config_name}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return 1            # dir doesn't exist yet
    nums = []
    for line in proc.stdout.splitlines():
        m = re.search(r"\brun_(\d+)\b", line)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def pull_run_back(config_name: str, run_n: int) -> Path:
    """Download the trained checkpoint + summary from the volume into the
    matching local `rl_bc/runs/<config_name>/run_<N>/` dir. Returns the local
    `best.pt` path.
    """
    remote_dir = f"/runs/{config_name}/run_{run_n}"
    local_dir = Path("rl_bc/runs") / config_name / f"run_{run_n}"
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists():
        import shutil
        shutil.rmtree(local_dir)
    subprocess.run(
        [*_MODAL_CMD, "volume", "get", "--force", VOLUME_NAME, remote_dir, str(local_dir.parent)],
        check=True,
    )
    return local_dir / 'best.pt'


# --------------------------------------------------------------------------- #
# Modal app / image / volume / remote function
# --------------------------------------------------------------------------- #


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO_ROOT,
        ignore=[
            "__pycache__", "*.pyc", ".git", ".vscode", ".pytest_cache",
            "rl_bc/cache", "rl_bc/runs", "rl_bc/_modal", "rl_bc/rollouts",
            "_internal", "*.exe", "ATC-Sim.exe",
            "human_data", "doc",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
_secrets = [modal.Secret.from_name(WANDB_SECRET_NAME)]


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    secrets=_secrets,
    gpu="A100-40GB",
    cpu=16,                      # match num_workers default — DataLoader is the bottleneck for h=64 MLP
    memory=32 * 1024,
    timeout=6 * 60 * 60,         # 200x repeat + 25 epochs needs ~3-4hr; pad for headroom
)
def train_remote(args: dict):
    """Run one bc_fm training run inside a Modal container.

    `args` keys forwarded from the local entrypoint:
      config_name, run_n, fold, n_folds, final, seed, epochs, rebuild_cache,
      use_wandb, num_workers.
    """
    import os
    import sys
    from pathlib import Path

    # Surface real CUDA tracebacks. Default async kernel launches make
    # asserts blame whichever kernel happens to sync, not the offender.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)

    from rl_bc.config import Config
    from rl_bc.configs import load_config
    from rl_bc.data import build_cache
    from rl_bc.train import train_one_fold  # dispatcher routes to cfg.family's train_one_run

    config_name = args["config_name"]
    run_n = int(args["run_n"])
    exp = load_config(config_name)

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    run_dir = vol_root / "runs" / exp.NAME / f"run_{run_n}"

    cfg = Config(
        name=exp.NAME,
        family=str(exp.FAMILY),
        fold=int(args.get("fold", 0)),
        n_folds=int(args.get("n_folds", 5)),
        final=bool(args.get("final", True)),
        final_val_fraction=float(exp.FINAL_VAL_FRACTION),
        early_stop_patience=int(exp.EARLY_STOP_PATIENCE),
        epochs=int(args.get("epochs", exp.EPOCHS)),
        batch_size=int(exp.BATCH_SIZE),
        lr=float(exp.LR),
        weight_decay=float(exp.WEIGHT_DECAY),
        warmup_epochs=int(exp.WARMUP_EPOCHS),
        grad_clip=float(exp.GRAD_CLIP),
        seed=int(args.get("seed", exp.SEED)),
        dropout=float(exp.DROPOUT),
        input_indices=tuple(exp.INPUT_INDICES),
        hidden=int(exp.HIDDEN),
        hdg_loss_weight=float(exp.HDG_LOSS_WEIGHT),
        alt_loss_weight=float(exp.ALT_LOSS_WEIGHT),
        spd_loss_weight=float(exp.SPD_LOSS_WEIGHT),
        fm_t_embed_dim=int(getattr(exp, "FM_T_EMBED_DIM", 16)),
        fm_n_steps=int(getattr(exp, "FM_N_STEPS", 10)),
        fm_noise_scale=float(getattr(exp, "FM_NOISE_SCALE", 1.0)),
        n_components=int(getattr(exp, "N_COMPONENTS", 4)),
        mixture_balance_weight=float(getattr(exp, "MIXTURE_BALANCE_WEIGHT", 0.0)),
        row_entropy_weight=float(getattr(exp, "ROW_ENTROPY_WEIGHT", 0.0)),
        row_entropy_target=float(getattr(exp, "ROW_ENTROPY_TARGET", 0.0)),
        focal_gamma=float(getattr(exp, "FOCAL_GAMMA", 0.0)),
        focal_threshold=float(getattr(exp, "FOCAL_THRESHOLD", 0.0)),
        focal_temperature=float(getattr(exp, "FOCAL_TEMPERATURE", 1.0)),
        heading_dropout_prob=float(getattr(exp, "HEADING_DROPOUT_PROB", 0.0)),
        alt_dropout_prob=float(getattr(exp, "ALT_DROPOUT_PROB", 0.0)),
        spd_dropout_prob=float(getattr(exp, "SPD_DROPOUT_PROB", 0.0)),
        dataset_repeat=int(getattr(exp, "DATASET_REPEAT", 1)),
        input_noise_heading_deg=float(getattr(exp, "INPUT_NOISE_HEADING_DEG", 0.0)),
        input_noise_alt_ft=float(getattr(exp, "INPUT_NOISE_ALT_FT", 0.0)),
        input_noise_spd_kt=float(getattr(exp, "INPUT_NOISE_SPD_KT", 0.0)),
        noise_clamp_alt_min_ft=float(getattr(exp, "NOISE_CLAMP_ALT_MIN_FT", 1000.0)),
        noise_clamp_alt_max_ft=float(getattr(exp, "NOISE_CLAMP_ALT_MAX_FT", 18000.0)),
        noise_clamp_spd_min_kt=float(getattr(exp, "NOISE_CLAMP_SPD_MIN_KT", 140.0)),
        noise_clamp_spd_max_kt=float(getattr(exp, "NOISE_CLAMP_SPD_MAX_KT", 280.0)),
        filter_loc_for_hdg=bool(getattr(exp, "FILTER_LOC_FOR_HDG", False)),
        train_sources=tuple(getattr(exp, "TRAIN_SOURCES", ())),
        cache_path=vol_root / "cache" / "bc_dataset.npz",
        run_dir=run_dir,
        num_workers=int(args.get("num_workers", 16)),
    )
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    human_data_dir = vol_root / "human_data"
    if not human_data_dir.exists():
        raise FileNotFoundError(f"{human_data_dir} not found; upload first.")

    if bool(args.get("rebuild_cache", False)) or not cfg.cache_path.exists():
        build_cache(cfg, human_dir=human_data_dir, cache_path=cfg.cache_path)
        volume.commit()

    use_wandb = bool(args.get("use_wandb", True))
    wandb_project = exp.WANDB_PROJECT
    wandb_run_name = f"{exp.NAME}_run_{run_n}"
    wandb_group = exp.NAME

    metric_hook = None
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            config={
                "config_name": cfg.name,
                "family": cfg.family,
                "run_n": run_n,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "seed": cfg.seed,
                "dropout": cfg.dropout,
                "hidden": cfg.hidden,
                "input_indices": list(cfg.input_indices),
                "fm_t_embed_dim": cfg.fm_t_embed_dim,
                "fm_n_steps": cfg.fm_n_steps,
                "fm_noise_scale": cfg.fm_noise_scale,
                "n_components": cfg.n_components,
                "filter_loc_for_hdg": cfg.filter_loc_for_hdg,
                "mixture_balance_weight": cfg.mixture_balance_weight,
                "row_entropy_weight": cfg.row_entropy_weight,
                "row_entropy_target": cfg.row_entropy_target,
                "warmup_epochs": cfg.warmup_epochs,
                "grad_clip": cfg.grad_clip,
                "early_stop_patience": cfg.early_stop_patience,
            },
        )

        def metric_hook(metrics: dict):
            step = metrics.get("step")
            payload = {k: v for k, v in metrics.items() if k != "step"}
            wandb.log(payload, step=step)

    try:
        summary = train_one_fold(cfg, metric_hook=metric_hook)
    finally:
        if use_wandb:
            import wandb
            wandb.finish()
        volume.commit()

    summary["run_n"] = run_n
    summary["wandb_project"] = wandb_project
    summary["wandb_run_name"] = wandb_run_name
    return summary


# --------------------------------------------------------------------------- #
# Remote eval — runs the parallel rollout sweep in one container with many CPUs.
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    cpu=32,                       # bumped from 16; pool autodetects ~16 workers
    memory=16 * 1024,
    timeout=2 * 60 * 60,
)
def eval_remote(args: dict) -> dict:
    """Run the parallel BC eval inside a Modal container, return summary+results."""
    import os
    import sys
    from pathlib import Path as PPath

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)

    from rl_bc.eval.runner import run_eval_parallel

    config_name = args["config_name"]
    vol_root = PPath(REMOTE_VOLUME_MOUNT)
    runs_root = vol_root / "runs" / config_name
    if not runs_root.exists():
        raise FileNotFoundError(f"no runs under {runs_root}/ on the volume")
    if args.get("run_n") is not None:
        ckpt_path = runs_root / f"run_{args['run_n']}" / "best.pt"
    else:
        runs = []
        for sub in runs_root.glob("run_*"):
            if (sub / "best.pt").exists():
                try:
                    runs.append((int(sub.name.split('_')[1]), sub))
                except (ValueError, IndexError):
                    continue
        if not runs:
            raise FileNotFoundError(f"no run_*/best.pt under {runs_root}")
        runs.sort()
        ckpt_path = runs[-1][1] / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

    summary, results = run_eval_parallel(
        ckpt_path=ckpt_path,
        cases=int(args.get("cases", 500)),
        max_steps=int(args.get("max_steps", 0)),    # 0 ⇒ use per-STAR caps
        warmup_wpts=int(args.get("warmup_wpts", 2)),
        runway=str(args.get("runway", '27')),
        airport=str(args.get("airport", 'test')),
        workers=int(args.get("workers", 16)),
        seed_base=int(args.get("seed_base", 0)),
        config_name=config_name,
    )
    return {"summary": summary, "results": results,
            "ckpt_path": str(ckpt_path)}


# --------------------------------------------------------------------------- #
# Big-CPU rollout — used by `bc_eval/run_bc_eval.py` to roll 200×6 episodes
# for a checkpoint and hand the per-step traces back for local scoring with
# `rl_ppo.eval_metrics`. Sized like the PPO rollout pool (rl_ppo/modal_config):
# cpu=64 ≈ 32 physical cores on EPYC after SMT, pool autodetects
# cpu_count // 2 = 32 workers so each lands on its own core. Separate from
# `eval_remote` (cpu=32/workers=16) so the bigger pool doesn't perturb the
# existing eval path.
# --------------------------------------------------------------------------- #

ROLLOUT_CPU = 64
ROLLOUT_WORKERS = 32


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    cpu=ROLLOUT_CPU,
    memory=16 * 1024,
    timeout=2 * 60 * 60,
)
def rollout_remote(args: dict) -> dict:
    """200×6 BC rollouts in one big-CPU container.

    Resolves the checkpoint from the volume by `config_name` (+ optional
    `run_n`, else latest run), runs the shared parallel eval, and returns
    {summary, results, ckpt_path}. `results` carry per-step a_traj/c_traj
    so the caller can build trajectories.npz and score it.
    """
    import os
    import sys as _sys
    from pathlib import Path as PPath

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REMOTE_REPO_ROOT)

    from rl_bc.eval.runner import run_eval_parallel

    config_name = args["config_name"]
    vol_root = PPath(REMOTE_VOLUME_MOUNT)
    runs_root = vol_root / "runs" / config_name
    if not runs_root.exists():
        raise FileNotFoundError(f"no runs under {runs_root}/ on the volume")

    if args.get("run_n"):
        ckpt_path = runs_root / f"run_{int(args['run_n'])}" / "best.pt"
    else:
        runs = []
        for sub in runs_root.glob("run_*"):
            if (sub / "best.pt").exists():
                try:
                    runs.append((int(sub.name.split('_')[1]), sub))
                except (ValueError, IndexError):
                    continue
        if not runs:
            raise FileNotFoundError(f"no run_*/best.pt under {runs_root}")
        runs.sort()
        ckpt_path = runs[-1][1] / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

    summary, results = run_eval_parallel(
        ckpt_path=ckpt_path,
        cases=int(args.get("cases", 200)),
        max_steps=int(args.get("max_steps", 0)),    # 0 ⇒ per-STAR caps
        warmup_wpts=int(args.get("warmup_wpts", 2)),
        runway=str(args.get("runway", '27')),
        airport=str(args.get("airport", 'test')),
        workers=int(args.get("workers", ROLLOUT_WORKERS)),
        seed_base=int(args.get("seed_base", 10_000)),
        config_name=config_name,
    )
    return {"summary": summary, "results": results,
            "ckpt_path": str(ckpt_path)}
