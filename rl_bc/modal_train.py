"""End-to-end Modal launcher.

One command per training run does everything:
  1. Detect new CSVs in `human_data/` and upload to the Modal volume.
  2. Allocate the next free run number (`<config>/run_N`).
  3. Train on a T4, logging to W&B project `<config.WANDB_PROJECT>`, group
     `<config.NAME>`, run name `<config.NAME>_run_N`.
  4. Pull the best checkpoint back to `rl_bc/runs/<config>/run_N/best.pt`.

Usage:
    modal run rl_bc/modal_train.py --config bc_fm_single
    modal run rl_bc/modal_train.py --config bc_fm_single --no-wandb
    modal run rl_bc/modal_train.py --config bc_fm_single --epochs 60 --seed 1
    modal run rl_bc/modal_train.py --config bc_fm_single --force-resync

--config is required so the command always names the model being trained.
"""

from rl_bc.configs import load_config
from rl_bc.modal_config import (
    app,
    next_run_number,
    pull_run_back,
    sync_local_data_to_modal_volume,
    train_remote,
)


@app.local_entrypoint()
def main(
    config: str,                    # required; e.g. bc_fm_single
    seed: int = 0,
    epochs: int = 0,                # 0 = use the config's value
    no_wandb: bool = False,
    force_resync: bool = False,     # re-upload human_data even if fingerprint matches
    num_workers: int = 16,
):
    exp = load_config(config)

    # 1. Sync local human_data → Modal volume (no-op if unchanged).
    synced = sync_local_data_to_modal_volume(force=force_resync)

    # 2. Allocate the next free run_N under runs/<NAME>/.
    run_n = next_run_number(exp.NAME)
    print(f"  → allocated {exp.NAME}_run_{run_n}")

    # 3. Train remotely.
    args = {
        "config_name": config,
        "run_n": run_n,
        "seed": seed,
        "final": True,
        "rebuild_cache": synced,        # force rebuild when CSVs changed
        "use_wandb": not no_wandb,
        "num_workers": num_workers,
    }
    if epochs > 0:
        args["epochs"] = epochs
    summary = train_remote.remote(args)
    print("\nsummary:", summary)

    # 4. Pull the checkpoint back to the local mirror.
    print(f"\n  → pulling {exp.NAME}/run_{run_n} back to rl_bc/runs/...")
    local_best = pull_run_back(exp.NAME, run_n)
    print(f"  ✓ checkpoint: {local_best}")
    print(f"\nrun with:  python -m rl_bc.bc_fm.watch")
    print(f"           python -m rl_bc.bc_fm.watch --run {run_n}")
