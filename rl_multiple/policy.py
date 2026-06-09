"""CombinedPolicy — frozen GMM (deterministic mode) + trainable Δ head.

Action accounting:

  PPO action (3-D):   Δ ∈ ℝ³  = (Δhdg_deg, Δalt_kft, Δspd_kt) — sampled
                                  from the delta head's Gaussian. PPO's
                                  log_prob / entropy are over THIS.
  Env action (4-D):   final = gmm_mode + Δ, expressed as
                              (sin θ, cos θ, alt_kft, spd_norm), where
                                  θ_final_deg = (atan2(s_g, c_g) + Δhdg) % 360
                                  alt_kft     = alt_g + Δalt_kft
                                  spd_norm    = spd_g + Δspd_kt / 100.0

The GMM is frozen — its parameters never enter the optimizer. The delta
head and critic are trained. Stochasticity at rollout time lives entirely
in the delta head's Gaussian sample.

Per-callsign frozen-noise pattern (same as bc_gmm Runtime): caller passes
a torch.Generator into `act()` re-seeded with a per-callsign hash every
tick. Both the GMM (argmax-mode is deterministic — no RNG actually used)
and the delta head's Normal.sample(generator=...) accept this.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from rl_bc.bc_gmm.model import BCActor, build_from_saved

from rl_multiple.critic import ValueNetwork
from rl_multiple.delta_head import DeltaHead
from rl_multiple.runtime import resolve_bc_seed_path


HDG_DIM = 0
ALT_DIM = 1
SPD_DIM = 2


def _combine_action(gmm_mode_4d: torch.Tensor,
                    delta_3d: torch.Tensor) -> torch.Tensor:
    """Add delta (3-D physical units) to the GMM's 4-D action.

    Inputs:
        gmm_mode_4d: (B, 4) = (sin θ, cos θ, alt_kft, spd_norm)
        delta_3d:    (B, 3) = (Δhdg_deg, Δalt_kft, Δspd_kt)

    Output:
        final 4-D action in the same units the env eats.
    """
    sin_g = gmm_mode_4d[..., 0]
    cos_g = gmm_mode_4d[..., 1]
    alt_g = gmm_mode_4d[..., 2]
    spd_g_norm = gmm_mode_4d[..., 3]
    d_hdg = delta_3d[..., HDG_DIM]
    d_alt = delta_3d[..., ALT_DIM]
    d_spd = delta_3d[..., SPD_DIM]

    # Heading: extract θ from sin/cos, add Δhdg, recompose.
    theta_g = torch.atan2(sin_g, cos_g)              # rad in (-π, π]
    theta_combined = theta_g + d_hdg * (math.pi / 180.0)
    sin_f = torch.sin(theta_combined)
    cos_f = torch.cos(theta_combined)

    alt_f = alt_g + d_alt
    # spd_norm = (kt - 200) / 100 → Δkt corresponds to Δspd_norm = Δkt / 100
    spd_f_norm = spd_g_norm + d_spd / 100.0

    return torch.stack([sin_f, cos_f, alt_f, spd_f_norm], dim=-1)


class CombinedPolicy(nn.Module):
    """Frozen GMM + DeltaHead + Critic."""

    def __init__(self, gmm: BCActor, delta_head: DeltaHead,
                 critic: ValueNetwork):
        super().__init__()
        self.gmm = gmm
        self.delta_head = delta_head
        self.critic = critic
        # Freeze GMM permanently.
        for p in self.gmm.parameters():
            p.requires_grad_(False)
        self.gmm.eval()

    def train(self, mode: bool = True):
        """Override: keep the GMM in eval mode regardless of self.train()
        toggles, since it's frozen and we never want its dropout to fire."""
        super().train(mode)
        self.gmm.eval()
        return self

    # ---------------------------------------------------------- #
    # Constructors
    # ---------------------------------------------------------- #

    @classmethod
    def from_ppo_ckpt(cls, ppo_ckpt_path: str | Path,
                      *,
                      bc_seed_path: str | Path | None = None,
                      density_n_bins: int = 36,
                      delta_hidden: int = 64,
                      value_hidden: int = 64,
                      value_dropout: float = 0.0,
                      delta_kwargs: dict | None = None,
                      device: str | torch.device = 'cpu',
                      ) -> 'CombinedPolicy':
        """Build the combined policy from a PPO checkpoint.

        Loads the actor_state into a fresh BCActor (sized via the BC
        seed referenced by the PPO ckpt's config), freezes it, then
        attaches a fresh delta head + critic.
        """
        device = torch.device(device)
        ppo_blob = torch.load(ppo_ckpt_path, map_location=device,
                              weights_only=False)
        if 'actor_state' not in ppo_blob:
            raise ValueError(
                f"{ppo_ckpt_path}: not a PPO ckpt "
                f"(no actor_state; keys={list(ppo_blob.keys())[:6]})"
            )
        seed_path = (Path(bc_seed_path) if bc_seed_path
                     else resolve_bc_seed_path(ppo_blob))
        seed_blob = torch.load(seed_path, map_location=device,
                               weights_only=False)
        gmm = build_from_saved(seed_blob['config']).to(device)
        gmm.load_state_dict(ppo_blob['actor_state'])
        gmm.eval()
        # Sanitize input_indices the same way rl_ppo.policy does — the
        # PPO env already slices the obs to the 7 columns the BC actor
        # was trained on, so the actor must not re-slice with its own
        # input_indices.
        in_features = gmm.encoder[0].in_features
        gmm.input_indices = list(range(in_features))
        gmm._idx_t = None

        ego_dim = in_features                          # 7
        full_dim = ego_dim + 2 * density_n_bins        # 7 + 72 = 79

        delta_kwargs = dict(delta_kwargs or {})
        delta_kwargs.setdefault('hidden', delta_hidden)
        delta_head = DeltaHead(input_dim=full_dim, **delta_kwargs).to(device)
        critic = ValueNetwork(input_dim=full_dim, hidden=value_hidden,
                              dropout=value_dropout).to(device)
        return cls(gmm, delta_head, critic)

    # ---------------------------------------------------------- #
    # State sync for mp workers (same pattern as rl_ppo.policy)
    # ---------------------------------------------------------- #

    def state_dict_split(self) -> dict:
        """Picklable dict shipped to workers each PPO iteration so they
        sample from the latest policy. GMM is excluded (frozen, never
        changes — workers loaded it from the same ckpt at init)."""
        return {
            'delta_head': {k: v.detach().cpu()
                           for k, v in self.delta_head.state_dict().items()},
            'critic': {k: v.detach().cpu()
                       for k, v in self.critic.state_dict().items()},
        }

    def load_state_split(self, state: dict,
                         device: str | torch.device = 'cpu') -> None:
        device = torch.device(device)
        self.delta_head.load_state_dict(
            {k: v.to(device) for k, v in state['delta_head'].items()})
        self.critic.load_state_dict(
            {k: v.to(device) for k, v in state['critic'].items()})

    def init_radar_head_from(self, ckpt_path: str | Path,
                              device: str | torch.device = 'cpu',
                              load_critic: bool = False) -> None:
        """Warm-start the radar head from a previously trained
        multi-PPO ckpt. Optimizer state is NOT loaded — caller builds
        fresh optimizers around the loaded params.

        `load_critic` defaults to **False** because the critic's
        learned V is biased when the reward structure changes between
        the source run and this one. A wrong-shaped warm V is worse
        than a zero-init cold V (which is unbiased but high-variance);
        cold critic learns the new reward structure in ~10-15 iters.

        Set `load_critic=True` only if you know the reward shape is
        unchanged (e.g. another Phase 2 → Phase 2 continuation).

        Use case: seed Phase 2 from `phase1_v1/best.pt`. Phase 1 only
        saw zero-density inputs, so the head's behavior on non-zero
        density is whatever its init left it at (mostly ~zero with σ
        carry-over). That's the right starting point — Phase 2 starts
        with a calibrated no-op response to quiet ticks.
        """
        device = torch.device(device)
        blob = torch.load(ckpt_path, map_location=device,
                          weights_only=False)
        if 'delta_head_state' not in blob:
            raise ValueError(
                f"{ckpt_path}: missing delta_head_state "
                f"(keys: {list(blob.keys())[:6]}...)"
            )
        self.delta_head.load_state_dict(blob['delta_head_state'])
        if load_critic and 'critic_state' in blob:
            self.critic.load_state_dict(blob['critic_state'])

    # ---------------------------------------------------------- #
    # GMM base — stochastic sample (matches single-plane PPO training).
    # ---------------------------------------------------------- #

    @torch.no_grad()
    def gmm_base(self, ego_obs: torch.Tensor,
                 generator: Optional[torch.Generator] = None,
                 deterministic: bool = False) -> torch.Tensor:
        """Frozen-GMM base action. Stochastic by default — picks a
        component via multinomial, then samples vMF (heading) +
        Gaussian (alt/spd) within it. This matches how the single-plane
        PPO was trained, so behavior at iter 0 of Phase 1 equals the
        ship's behavior modulo the small delta-head Gaussian noise.

        With a re-seeded `generator` per tick, the sampled base is
        reproducible per (callsign, state) — same per-callsign
        "frozen noise" pattern as bc_gmm Runtime.

        `deterministic=True` falls back to the argmax-component mean —
        useful for offline eval / probing, not training.
        """
        if ego_obs.dim() == 1:
            ego_obs = ego_obs.unsqueeze(0)
        c = self.gmm.encode(ego_obs)
        return self.gmm.sample(c, generator=generator,
                               deterministic=deterministic)

    # ---------------------------------------------------------- #
    # Rollout-side: sample delta + compose final action
    # ---------------------------------------------------------- #

    @torch.no_grad()
    def act(self, ego_obs: torch.Tensor, full_obs: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            deterministic_base: bool = False) -> dict:
        """Sample one delta per row and return everything the rollout needs.

        Inputs:
            ego_obs:   (B, 7)  — input to the frozen GMM
            full_obs:  (B, 79) — input to the delta head & critic
            generator: per-callsign RNG. Used for BOTH the GMM sample
                       (component pick + vMF/Gaussian draws) and the
                       delta head's Gaussian → reproducible per
                       callsign across re-runs.
            deterministic_base: GMM uses argmax-component mean instead
                       of sampling. Default False (stochastic, matches
                       training distribution).

        Returns dict:
            action_delta: (B, 3) physical-units Δ — the PPO action
            action_final: (B, 4) gmm_base + delta — the env action
            log_prob:     (B,)   under the delta head's Gaussian
                                  (the GMM base is exogenous to PPO;
                                  its randomness does NOT enter log_prob)
            value:        (B,)   V(full_obs)
        """
        if ego_obs.dim() == 1:
            ego_obs = ego_obs.unsqueeze(0)
        if full_obs.dim() == 1:
            full_obs = full_obs.unsqueeze(0)

        # GMM sample first — uses some bits from the generator. Delta
        # head's Gaussian draws the rest. Reproducible per generator state.
        base = self.gmm_base(ego_obs, generator=generator,
                             deterministic=deterministic_base)

        mu, sigma = self.delta_head(full_obs)
        eps = torch.randn(mu.shape, generator=generator,
                          device=mu.device, dtype=mu.dtype)
        delta = mu + sigma * eps
        log_prob = self._gaussian_log_prob(delta, mu, sigma)

        final = _combine_action(base, delta)
        value = self.critic(full_obs)
        return {
            'action_delta': delta,
            'action_final': final,
            'log_prob': log_prob,
            'value': value,
        }

    # ---------------------------------------------------------- #
    # Update-side: re-score stored deltas
    # ---------------------------------------------------------- #

    def evaluate_actions(self, full_obs: torch.Tensor,
                         action_delta: torch.Tensor) -> dict:
        """Re-score the stored 3-D delta under the *current* delta head.
        Gradient flows through delta_head + critic; GMM is detached
        (frozen at construction)."""
        mu, sigma = self.delta_head(full_obs)
        log_prob = self._gaussian_log_prob(action_delta, mu, sigma)
        # Exact diagonal-Gaussian entropy: H = 0.5 * sum(log(2πe σ²))
        entropy = 0.5 * (1.0 + math.log(2.0 * math.pi)) * sigma.shape[-1] \
            + sigma.log().sum(dim=-1)
        value = self.critic(full_obs)
        return {'log_prob': log_prob, 'entropy': entropy, 'value': value}

    @staticmethod
    def _gaussian_log_prob(x: torch.Tensor,
                           mu: torch.Tensor,
                           sigma: torch.Tensor) -> torch.Tensor:
        """Diagonal-Gaussian log-density, summed over the action dim."""
        var = sigma * sigma
        log_two_pi = math.log(2.0 * math.pi)
        per_dim = -0.5 * (((x - mu) ** 2) / var
                          + var.log()
                          + log_two_pi)
        return per_dim.sum(dim=-1)
