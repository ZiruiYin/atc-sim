"""Per-experiment config loader.

Each config is a Python module under `rl_bc/configs/` that defines a `NAME`,
hyperparams, and a `WANDB_PROJECT`. Today there's one (`bc_fm_single`); the
loader is generic so adding future families (e.g. `bc_gmm_single`) is a
single new file.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent


def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.py") if p.stem != "__init__")


def load_config(name: str) -> types.ModuleType:
    try:
        return importlib.import_module(f"rl_bc.configs.{name}")
    except ImportError as e:
        raise ValueError(
            f"unknown config '{name}'. Available: {', '.join(list_configs())}"
        ) from e
