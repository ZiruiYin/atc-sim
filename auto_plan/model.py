"""Self-contained GMM-vMF policy model for the AUTO planner.

Ported verbatim from the rl branch's `rl_bc/bc_gmm/model.py` so `auto_plan`
has no dependency on the training package. Per component k:

    angle      ~ VonMises(mu_k, kappa_k)              (1 angular dim)
    alt_std    ~ N(mu_alt_k, sigma_alt_k)             (standardized)
    spd_std    ~ N(mu_spd_k, sigma_spd_k)
    weight     ~ softmax(logits)_k

External action is 4-D (sin theta, cos theta, alt_kft, spd_norm). The full
architecture (encoder 7->64->64, 4 mixture components) is recovered from the
checkpoint's `actor_state` tensor shapes; `policy_config.json` carries the
input-feature indices and the input standardizer the checkpoint was trained
with.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_2PI = math.log(2.0 * math.pi)

# Structural policy bounds enforced inside gmm_params via softplus reparam:
#   sigma >= exp(-2) ~ 0.135 (standardized);  kappa <= exp(4) ~ 55 (>= ~8 deg).
LOG_STD_FLOOR = -2.0
LOG_KAPPA_CEILING = 4.0


def _trunk(in_d: int, hid: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, hid), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid, hid),   nn.GELU(), nn.Dropout(dropout),
    )


def _vmf_sample(mu: torch.Tensor, kappa: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Vectorized von Mises sampler (Wood 1994). `mu`, `kappa` shape (B,);
    returns (B,) angles wrapped to [-pi, pi]."""
    B = mu.shape[0]
    device = mu.device
    dtype = mu.dtype
    PI = math.pi

    kappa_safe = kappa.clamp(min=1e-3)
    a = 1.0 + torch.sqrt(1.0 + 4.0 * kappa_safe ** 2)
    b = (a - torch.sqrt(2.0 * a)) / (2.0 * kappa_safe)
    r = (1.0 + b ** 2) / (2.0 * b)

    result = torch.zeros(B, device=device, dtype=dtype)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(50):
        if done.all():
            break
        u1 = torch.rand(B, device=device, dtype=dtype, generator=generator)
        u2 = torch.rand(B, device=device, dtype=dtype, generator=generator)
        u3 = torch.rand(B, device=device, dtype=dtype, generator=generator)
        z = torch.cos(PI * u1)
        f = (1.0 + r * z) / (r + z)
        c_val = kappa_safe * (r - f)
        log_u2 = torch.log(u2.clamp(min=1e-30))
        accept = (c_val * (2.0 - c_val) > u2) | (
            torch.log(c_val.clamp(min=1e-30)) + 1.0 - c_val > log_u2
        )
        sign = torch.where(u3 > 0.5, torch.ones_like(u3), -torch.ones_like(u3))
        theta = sign * torch.acos(f.clamp(-1.0, 1.0))
        new = accept & ~done
        result = torch.where(new, mu + theta, result)
        done = done | accept

    if not bool(done.all()):
        result = torch.where(done, result, mu)
    return torch.atan2(torch.sin(result), torch.cos(result))


class BCActor(nn.Module):
    """Joint mixture of (vMF heading, Gaussian alt, Gaussian spd)."""

    ACTION_DIM = 4   # external order: (sin theta, cos theta, alt_kft, spd_norm)

    def __init__(self,
                 input_indices: tuple[int, ...] = (0, 1, 2),
                 hidden: int = 64,
                 n_components: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_indices = list(input_indices)
        self._idx_t: torch.Tensor | None = None
        self.hidden = hidden
        self.K = int(n_components)

        self.encoder = _trunk(len(self.input_indices), hidden, dropout)

        self.head_logits = nn.Linear(hidden, self.K)
        self.head_angle_xy = nn.Linear(hidden, self.K * 2)
        self.head_log_kappa = nn.Linear(hidden, self.K)
        self.head_alt_mean = nn.Linear(hidden, self.K)
        self.head_alt_log_std = nn.Linear(hidden, self.K)
        self.head_spd_mean = nn.Linear(hidden, self.K)
        self.head_spd_log_std = nn.Linear(hidden, self.K)

        self.register_buffer('target_mean',
                             torch.zeros(self.ACTION_DIM, dtype=torch.float32))
        self.register_buffer('target_std',
                             torch.ones(self.ACTION_DIM, dtype=torch.float32))

    def _idx(self, x: torch.Tensor) -> torch.Tensor:
        if self._idx_t is None or self._idx_t.device != x.device:
            self._idx_t = torch.tensor(self.input_indices,
                                       dtype=torch.long, device=x.device)
        return self._idx_t

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.index_select(dim=-1, index=self._idx(x)))

    def gmm_params(self, c: torch.Tensor):
        B = c.shape[0]
        K = self.K
        logits = self.head_logits(c)
        angle_xy = self.head_angle_xy(c).view(B, K, 2)
        mu_angle = torch.atan2(angle_xy[..., 0], angle_xy[..., 1])
        raw_log_kappa = self.head_log_kappa(c)
        log_kappa = LOG_KAPPA_CEILING - F.softplus(LOG_KAPPA_CEILING - raw_log_kappa)
        kappa = log_kappa.exp()
        alt_mean = self.head_alt_mean(c)
        spd_mean = self.head_spd_mean(c)
        raw_alt_log_std = self.head_alt_log_std(c)
        raw_spd_log_std = self.head_spd_log_std(c)
        alt_log_std = LOG_STD_FLOOR + F.softplus(raw_alt_log_std - LOG_STD_FLOOR)
        spd_log_std = LOG_STD_FLOOR + F.softplus(raw_spd_log_std - LOG_STD_FLOOR)
        return logits, mu_angle, kappa, alt_mean, alt_log_std, spd_mean, spd_log_std

    @torch.no_grad()
    def sample(self, c: torch.Tensor,
               generator: torch.Generator | None = None,
               deterministic: bool = False) -> torch.Tensor:
        """Sample one 4-D action in physical units (sin, cos, alt_kft, spd_norm)."""
        logits, mu_angle, kappa, alt_mean, alt_log_std, spd_mean, spd_log_std = \
            self.gmm_params(c)
        B = c.shape[0]
        idx_b = torch.arange(B, device=c.device)
        if deterministic:
            comp = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            comp = torch.multinomial(probs, num_samples=1,
                                     generator=generator).squeeze(-1)

        mu_k = mu_angle[idx_b, comp]
        kappa_k = kappa[idx_b, comp]
        alt_m_k = alt_mean[idx_b, comp]
        alt_s_k = alt_log_std[idx_b, comp].exp()
        spd_m_k = spd_mean[idx_b, comp]
        spd_s_k = spd_log_std[idx_b, comp].exp()

        if deterministic:
            angle = mu_k
            alt_std = alt_m_k
            spd_std = spd_m_k
        else:
            angle = _vmf_sample(mu_k, kappa_k, generator=generator)
            noise = torch.randn(B, 2, device=c.device, dtype=c.dtype,
                                generator=generator)
            alt_std = alt_m_k + alt_s_k * noise[:, 0]
            spd_std = spd_m_k + spd_s_k * noise[:, 1]

        alt_kft = alt_std * self.target_std[2] + self.target_mean[2]
        spd_norm = spd_std * self.target_std[3] + self.target_mean[3]
        return torch.stack([torch.sin(angle), torch.cos(angle), alt_kft, spd_norm],
                           dim=-1)


def build_actor(input_indices, hidden=64, n_components=4, dropout=0.1) -> BCActor:
    return BCActor(input_indices=tuple(input_indices), hidden=int(hidden),
                   n_components=int(n_components), dropout=float(dropout))
