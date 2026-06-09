"""bc_gmm model: per-component mixture of von Mises (heading) and Gaussians
(altitude, speed). Diagonal across the three action channels.

Why not diagonal Gaussian on (sin θ, cos θ): a narrow arc of angles on the
unit circle is a *curve* in (sin, cos) space, not an axis-aligned box. An
independent Gaussian in each of sin and cos can only approximate this with
σ wide enough to permit "off the circle" mass — which means sampling
produces angles that are tens of degrees noisy even when the demonstrator
was extremely concentrated. The old version of this model exhibited exactly
that failure.

vMF concentration κ directly parameterizes how tight the angle distribution
is around the mean direction μ; sampled angles stay on the circle by
construction. Alt and spd remain diagonal Gaussians (standardized so the
NLL gradient is well-scaled).

Per component k:
    angle      ~ VonMises(μ_k, κ_k)                  (1 angular dim)
    alt_std    ~ N(μ_alt_k, σ_alt_k)                 (in standardized space)
    spd_std    ~ N(μ_spd_k, σ_spd_k)
    weight     ~ softmax(logits)_k

External interface is 4-D (sin θ, cos θ, alt_kft, spd_norm) to match
bc_fm's contract — `sample()` returns those four values in physical units,
`log_prob()` takes them too. The (sin, cos) pair is collapsed into an
angle internally before the vMF density evaluation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_2PI = math.log(2.0 * math.pi)

# Structural policy bounds, enforced INSIDE gmm_params via smooth
# softplus reparameterization (not as soft penalty terms in the loss).
# The reparam guarantees the bound is satisfied for any unconstrained head
# output, while keeping gradient flowing smoothly everywhere — unlike the
# old clamp + soft-penalty combo, which let NLL park σ → 0 / κ → ∞ on
# whichever components carried real data, complying only on dead ones.
#
# Effective consequences:
#   σ ≥ exp(-2)  ≈ 0.135  in standardized units (~13.5% of action std)
#   κ ≤ exp(4)   ≈ 55             → angular std ≥ ~8°
LOG_STD_FLOOR = -2.0
LOG_KAPPA_CEILING = 4.0


def _trunk(in_d: int, hid: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, hid), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid, hid),   nn.GELU(), nn.Dropout(dropout),
    )


def _vmf_sample(mu: torch.Tensor, kappa: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Vectorized von Mises sampler (Wood 1994), accepting a torch.Generator
    for per-aircraft reproducibility. `mu`, `kappa` shape (B,); returns (B,).
    Angles are wrapped to [-π, π].
    """
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

    # Fallback for any unaccepted samples → μ.
    if not bool(done.all()):
        result = torch.where(done, result, mu)
    # Wrap to [-π, π].
    return torch.atan2(torch.sin(result), torch.cos(result))


class BCActor(nn.Module):
    """Joint mixture of (vMF heading, Gaussian alt, Gaussian spd)."""

    ACTION_DIM = 4   # external dim order: (sin θ, cos θ, alt_kft, spd_norm)

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

        # Mixture weights.
        self.head_logits = nn.Linear(hidden, self.K)
        # vMF μ via 2D atan2: keeps the mean on the circle without modular wraps.
        self.head_angle_xy = nn.Linear(hidden, self.K * 2)
        # vMF κ via softplus on the clamped log-output.
        self.head_log_kappa = nn.Linear(hidden, self.K)
        # Gaussian alt + spd (standardized).
        self.head_alt_mean = nn.Linear(hidden, self.K)
        self.head_alt_log_std = nn.Linear(hidden, self.K)
        self.head_spd_mean = nn.Linear(hidden, self.K)
        self.head_spd_log_std = nn.Linear(hidden, self.K)

        # Small init for angle xy, zero for the rest → starts near
        # uniform circle / unit-variance Gaussian.
        nn.init.normal_(self.head_angle_xy.weight, std=0.05)
        nn.init.zeros_(self.head_angle_xy.bias)
        for h in (self.head_logits, self.head_log_kappa,
                  self.head_alt_mean, self.head_alt_log_std,
                  self.head_spd_mean, self.head_spd_log_std):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        # Standardization buffers: shape (4,) for backward compatibility with
        # the bc_fm interface (set_target_stats / unstandardize_sample callers).
        # vMF doesn't use indices 0/1 (sin, cos) — only alt (2) and spd (3) are
        # actually standardized inside this model.
        self.register_buffer('target_mean',
                             torch.zeros(self.ACTION_DIM, dtype=torch.float32))
        self.register_buffer('target_std',
                             torch.ones(self.ACTION_DIM, dtype=torch.float32))

    # ---- standardization (alt/spd only) ---- #

    def set_target_stats(self, mean, std) -> None:
        m = torch.as_tensor(mean, dtype=torch.float32, device=self.target_mean.device)
        s = torch.as_tensor(std,  dtype=torch.float32, device=self.target_std.device)
        s = torch.where(s < 1e-6, torch.ones_like(s), s)
        self.target_mean.copy_(m)
        self.target_std.copy_(s)

    def standardize_target(self, y: torch.Tensor) -> torch.Tensor:
        """No-op pass-through. log_prob handles internal standardization for
        alt/spd; the angle dims are consumed via atan2(sin, cos) directly.
        """
        return y

    def unstandardize_sample(self, x: torch.Tensor) -> torch.Tensor:
        """No-op pass-through. `sample()` already returns physical units."""
        return x

    # ---- encode + GMM-vMF parameters ---- #

    def _idx(self, x: torch.Tensor) -> torch.Tensor:
        if self._idx_t is None or self._idx_t.device != x.device:
            self._idx_t = torch.tensor(self.input_indices,
                                       dtype=torch.long, device=x.device)
        return self._idx_t

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.index_select(dim=-1, index=self._idx(x)))

    def gmm_params(self, c: torch.Tensor):
        """Returns the 7-tuple of per-component parameters:
        (logits, mu_angle, kappa, alt_mean_std, alt_log_std,
         spd_mean_std, spd_log_std) all shaped (B, K).

        σ and κ are reparameterized so the structural bounds are
        IMPOSSIBLE to violate, regardless of head output:

            log σ = LOG_STD_FLOOR + softplus(raw - LOG_STD_FLOOR)
                    ≥ LOG_STD_FLOOR   for all raw, smooth gradient.
            log κ = LOG_KAPPA_CEIL - softplus(LOG_KAPPA_CEIL - raw)
                    ≤ LOG_KAPPA_CEIL  for all raw, smooth gradient.

        Replaces the prior hard-clamp + soft-penalty combo, which the
        optimizer gamed by complying on unused components and parking
        the dominant one at the clamp boundary.
        """
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

    # ---- mixture diagnostics / regularizers ---- #

    def mixture_balance_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Scalar regularizer that fights mode collapse.

        Returns `log(K) - H(π̄)`, where `π̄ = E_batch[π(x)]` is the
        batch-averaged mixture distribution. Minimum 0 when batch-average
        usage is uniform across components; max `log(K)` when all rows
        collapse onto the same component.
        """
        logits = self.head_logits(c)
        pi = torch.softmax(logits, dim=-1)                   # (B, K)
        mean_pi = pi.mean(dim=0).clamp(min=1e-8)             # (K,)
        H = -(mean_pi * mean_pi.log()).sum()
        return torch.tensor(math.log(self.K), device=c.device,
                            dtype=c.dtype) - H

    # ---- training-time log-likelihood ---- #

    def log_prob_from_params(self, target_4d: torch.Tensor,
                             params: tuple,
                             keep_dims: torch.Tensor | None = None
                             ) -> torch.Tensor:
        """Same as `log_prob`, but reuses pre-computed `gmm_params(c)` output
        to avoid running the heads twice when the caller also needs the raw
        per-row parameters (for entropy or floor/ceiling penalties)."""
        logits, mu_angle, kappa, alt_mean, alt_log_std, spd_mean, spd_log_std = params

        target_angle = torch.atan2(target_4d[:, 0], target_4d[:, 1])         # (B,)
        target_alt = target_4d[:, 2]
        target_spd = target_4d[:, 3]
        alt_std_target = (target_alt - self.target_mean[2]) / self.target_std[2]
        spd_std_target = (target_spd - self.target_mean[3]) / self.target_std[3]

        # vMF log-density: κ cos(θ - μ) - log(2π) - log(I_0(κ))
        # log I_0(κ) = log I_0e(κ) + κ for numerical stability at large κ.
        log_i0 = torch.log(torch.special.i0e(kappa)) + kappa                  # (B, K)
        log_p_angle = (kappa * torch.cos(target_angle.unsqueeze(-1) - mu_angle)
                       - _LOG_2PI - log_i0)                                   # (B, K)

        alt_diff = alt_std_target.unsqueeze(-1) - alt_mean
        log_p_alt = (-0.5 * (alt_diff / alt_log_std.exp()) ** 2
                     - alt_log_std - 0.5 * _LOG_2PI)
        spd_diff = spd_std_target.unsqueeze(-1) - spd_mean
        log_p_spd = (-0.5 * (spd_diff / spd_log_std.exp()) ** 2
                     - spd_log_std - 0.5 * _LOG_2PI)

        if keep_dims is not None:
            log_p_angle = log_p_angle * keep_dims[:, 0:1]
            log_p_alt = log_p_alt * keep_dims[:, 2:3]
            log_p_spd = log_p_spd * keep_dims[:, 3:4]

        log_p_per_comp = (log_p_angle + log_p_alt + log_p_spd
                          + torch.log_softmax(logits, dim=-1))
        return torch.logsumexp(log_p_per_comp, dim=-1)                        # (B,)

    def log_prob(self, target_4d: torch.Tensor, c: torch.Tensor,
                 keep_dims: torch.Tensor | None = None) -> torch.Tensor:
        """log p(target | x) under the vMF·N·N mixture.

        `target_4d` is (B, 4) in raw units: (sin θ, cos θ, alt_kft, spd_norm).
        `keep_dims` is an optional (B, 4) weight matrix; index 0 (the sin
        column) is treated as the angle keep (sin and cos are paired into
        one angular channel), indices 2/3 are the alt/spd keeps.
        """
        return self.log_prob_from_params(target_4d, self.gmm_params(c),
                                         keep_dims=keep_dims)

    # ---- inference-time sampling ---- #

    @torch.no_grad()
    def sample(self, c: torch.Tensor,
               generator: torch.Generator | None = None,
               deterministic: bool = False) -> torch.Tensor:
        """Sample one 4-D action in physical units (sin θ, cos θ, alt_kft, spd_norm).

        Component selection via Categorical(softmax(logits)) accepts the
        `generator`; the vMF sampler does too (custom Wood implementation).
        `deterministic=True` returns the dominant-component (μ_angle, μ_alt, μ_spd).
        """
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


def build_from_saved(saved_cfg: dict) -> BCActor:
    """Reconstruct from a checkpoint's config dict. Standardization buffers
    restore automatically via state_dict.
    """
    return BCActor(
        input_indices=tuple(saved_cfg.get('input_indices', (0, 1, 2))),
        hidden=int(saved_cfg.get('hidden', 64)),
        n_components=int(saved_cfg.get('n_components', 4)),
        dropout=float(saved_cfg.get('dropout', 0.1)),
    )
