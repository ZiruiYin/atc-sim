"""Critic / value network.

Architectural mirror of the BC actor's encoder: 2-layer MLP, hidden=64,
GELU. Input is the same standardized observation that the actor sees —
size matches the BC actor's `input_indices` (3-D for bc_gmm_single,
7-D for bc_gmm_single_full). Output is a scalar V(s) estimate.

Trained from scratch (no BC initialization). Separate from the actor —
sharing the encoder makes value-loss and policy-loss gradients fight on
the same parameters, which slows learning and tends to blow up the
critic. Tiny model, separate weights is cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int = 7, hidden: int = 64,
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
        # Tiny init on the head so V starts near 0 (rewards are ±1).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3) standardized; returns (B,) scalar value."""
        return self.net(obs).squeeze(-1)
