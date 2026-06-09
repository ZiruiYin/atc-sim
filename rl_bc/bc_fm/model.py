"""bc_fm model: joint CFM over (sin θ, cos θ, alt_std, spd_std).

One position-only encoder + one 4-D velocity field. At inference an Euler ODE
turns N(0, σ²·I) noise into a 4-D action vector; un-standardize to physical
units and you have hdg / alt / spd in one shot.

Per-dim target standardization buffers (`target_mean`, `target_std`) are
populated at train time via `set_target_stats(...)` and ride along with the
state_dict so inference automatically de-standardizes the right way.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rl_bc.config import N_FEATURES  # noqa: F401 (kept for cross-module signature parity)


def _trunk(in_d: int, hid: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, hid), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid, hid),   nn.GELU(), nn.Dropout(dropout),
    )


class BCActor(nn.Module):
    """Joint flow-matching actor for hdg + alt + spd, all coupled.

    Target dim order:
        0  sin(target_heading)
        1  cos(target_heading)
        2  (target_altitude / 1000)        — kft
        3  (target_airspeed - 200) / 100   — normalized kt
    """

    ACTION_DIM = 4

    def __init__(self,
                 input_indices: tuple[int, ...] = (0, 1, 2),
                 hidden: int = 64,
                 dropout: float = 0.1,
                 fm_t_embed_dim: int = 16,
                 fm_noise_scale: float = 1.0):
        super().__init__()
        self.input_indices = list(input_indices)
        self._idx_t: torch.Tensor | None = None
        self.hidden = hidden
        self.fm_t_embed_dim = fm_t_embed_dim
        self.fm_noise_scale = float(fm_noise_scale)

        self.encoder = _trunk(len(self.input_indices), hidden, dropout)
        vel_in = self.ACTION_DIM + fm_t_embed_dim + hidden
        self.velocity = nn.Sequential(
            nn.Linear(vel_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.ACTION_DIM),
        )

        self.register_buffer('target_mean',
                             torch.zeros(self.ACTION_DIM, dtype=torch.float32))
        self.register_buffer('target_std',
                             torch.ones(self.ACTION_DIM, dtype=torch.float32))

    # ---- target standardization ---- #

    def set_target_stats(self, mean, std) -> None:
        m = torch.as_tensor(mean, dtype=torch.float32, device=self.target_mean.device)
        s = torch.as_tensor(std,  dtype=torch.float32, device=self.target_std.device)
        s = torch.where(s < 1e-6, torch.ones_like(s), s)
        self.target_mean.copy_(m)
        self.target_std.copy_(s)

    def standardize_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def unstandardize_sample(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.target_std + self.target_mean

    # ---- flow-matching primitives ---- #

    def _idx(self, x: torch.Tensor) -> torch.Tensor:
        if self._idx_t is None or self._idx_t.device != x.device:
            self._idx_t = torch.tensor(self.input_indices,
                                       dtype=torch.long, device=x.device)
        return self._idx_t

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.fm_t_embed_dim // 2
        device, dtype = t.device, t.dtype
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype)
            / max(1, half - 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.index_select(dim=-1, index=self._idx(x)))

    def velocity_field(self, x_t: torch.Tensor, t: torch.Tensor,
                       c: torch.Tensor) -> torch.Tensor:
        return self.velocity(torch.cat([x_t, self._t_embed(t), c], dim=-1))

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n_steps: int = 10,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample one 4-D *standardized* action vector. Caller un-standardizes."""
        B = c.shape[0]
        x = torch.randn(B, self.ACTION_DIM, device=c.device, dtype=c.dtype,
                        generator=generator) * self.fm_noise_scale
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((B,), k * dt, device=c.device, dtype=c.dtype)
            x = x + self.velocity_field(x, t, c) * dt
        return x


def build_from_saved(saved_cfg: dict) -> BCActor:
    """Reconstruct from a checkpoint's config dict. Target stats live in the
    state_dict buffers so they restore automatically on `model.load_state_dict`.
    """
    return BCActor(
        input_indices=tuple(saved_cfg.get('input_indices', (0, 1, 2))),
        hidden=int(saved_cfg.get('hidden', 64)),
        dropout=float(saved_cfg.get('dropout', 0.1)),
        fm_t_embed_dim=int(saved_cfg.get('fm_t_embed_dim', 16)),
        fm_noise_scale=float(saved_cfg.get('fm_noise_scale', 1.0)),
    )
