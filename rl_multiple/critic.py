"""Critic for multi-plane PPO.

Identical architecture to `rl_ppo.critic.ValueNetwork` — 2-layer MLP,
hidden 64, GELU — but takes the wider 79-D observation
(ego_7 + density_36 + density_delta_36) as input.

Trained from scratch. Separate from actor/delta-head to keep value and
policy gradients from fighting on the same parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int = 79, hidden: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden), nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        # Zero-init head so V starts near 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 79) standardized; returns (B,) scalar value."""
        return self.net(obs).squeeze(-1)
