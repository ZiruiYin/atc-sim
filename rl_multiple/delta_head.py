"""Delta head — trainable correction module on top of the frozen GMM.

Input:  79-D state_full = concat([ego_7, density_now_36, density_delta_36])
Output: 3-D Gaussian over physical command deltas:
            μ = (Δhdg_deg, Δalt_kft, Δspd_kt), tanh-clamped to per-dim limits
            σ = exp(log_σ_clamped), per-dim independent

Combined sim action = frozen_GMM_mode + (sampled Δ from this head).

Zero-init on the mean head means at training iter 0 the head outputs
μ = (0, 0, 0) regardless of input → combined action = pure GMM mode
exactly. Phase 1 starts from a verified-good baseline.

PPO treats the 3-D Δ as the action; log_prob / entropy come from the
standard diagonal-Gaussian formulas.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Default physical clamps and sigma range. Overridable via constructor
# args so MultiPPOConfig can drive them.
DEFAULT_HDG_CLAMP_DEG = 30.0
DEFAULT_ALT_CLAMP_KFT = 1.0
DEFAULT_SPD_CLAMP_KT = 30.0
DEFAULT_LOG_SIGMA_INIT = -1.5     # σ ≈ 0.22 (in same physical-unit scale)
DEFAULT_LOG_SIGMA_MIN = -3.5
DEFAULT_LOG_SIGMA_MAX = 0.0


class DeltaHead(nn.Module):
    def __init__(self,
                 input_dim: int = 79,
                 hidden: int = 64,
                 hdg_clamp_deg: float = DEFAULT_HDG_CLAMP_DEG,
                 alt_clamp_kft: float = DEFAULT_ALT_CLAMP_KFT,
                 spd_clamp_kt: float = DEFAULT_SPD_CLAMP_KT,
                 log_sigma_init: float = DEFAULT_LOG_SIGMA_INIT,
                 log_sigma_min: float = DEFAULT_LOG_SIGMA_MIN,
                 log_sigma_max: float = DEFAULT_LOG_SIGMA_MAX):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, 3)
        self.log_sigma_head = nn.Linear(hidden, 3)

        # CRITICAL: zero-init last layers so:
        #   - mu_head outputs raw 0 → tanh(0) * scale = 0 → no delta at init
        #   - log_sigma_head outputs `log_sigma_init` everywhere → σ ≈ 0.22
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, log_sigma_init)

        # Stored as buffers so they ride along with state_dict / device moves.
        self.register_buffer(
            'mu_scale',
            torch.tensor([hdg_clamp_deg, alt_clamp_kft, spd_clamp_kt],
                         dtype=torch.float32),
        )
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, input_dim) → (mu, sigma) each shape (B, 3).

        `mu` is in physical units (deg, kft, kt) and bounded to
        ±(hdg_clamp_deg, alt_clamp_kft, spd_clamp_kt) by tanh.
        `sigma` is in the same physical-unit scale (NOT normalized).
        """
        h = self.encoder(x)
        mu = torch.tanh(self.mu_head(h)) * self.mu_scale
        log_sigma = self.log_sigma_head(h).clamp(self.log_sigma_min,
                                                  self.log_sigma_max)
        sigma = log_sigma.exp()
        return mu, sigma
