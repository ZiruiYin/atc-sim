"""Multi-plane PPO env (Phase 1: single-plane wrapper).

Extends `rl_ppo.env.PPOEnv` by augmenting the observation with the
36-bin traffic density (`now`) and its single-tick temporal delta.

Phase 1 spawns one plane at a time → density is identically zero on
every step. The delta head still receives the full 79-D input and
learns to ignore the zero-density slot (zero-init last layer keeps
its output ≈ 0 at training start, so behavior matches the frozen GMM
ship baseline).

Phase 2 will swap the sim for `spawn_single=False` with neighbors
populated; the density slot then carries real signal and the head's
job becomes meaningful.

Observation:
    obs[0:7]   ego state (a_nm, c_nm, d_thr, alt_kft, spd_norm,
                            sin h, cos h) — standardized as in PPOEnv
    obs[7:43]  density_now (36 bins, ego-relative bearing)
    obs[43:79] density_delta = now - prev_now (zero on the first tick
                                                 of each episode)

The action interface is unchanged: env.step takes the same 4-D
(sin θ, cos θ, alt_kft, spd_norm) command that PPOEnv eats. Combining
the GMM mode with the delta head's output to produce that 4-D command
is `CombinedPolicy`'s responsibility (rollout-side glue).
"""
from __future__ import annotations

import numpy as np

from rl_ppo.env import PPOEnv, StepResult     # re-exported for convenience

from rl_multiple.density import N_BINS, CUTOFF_NM, build_density


# Re-export so callers don't need to know about rl_ppo internals.
__all__ = ['MultiPPOEnv', 'StepResult']


class MultiPPOEnv(PPOEnv):
    """Single-plane PPOEnv extended with a traffic-density observation."""

    DENSITY_DIM = N_BINS                       # 36

    def __init__(self, *args,
                 density_cutoff_nm: float = CUTOFF_NM,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._ego_obs_dim = self.obs_dim
        self.obs_dim = self._ego_obs_dim + 2 * self.DENSITY_DIM
        self._density_cutoff_nm = float(density_cutoff_nm)
        # Per-episode rolling buffer for density delta. Reset at episode
        # start. None ⇒ first observation, delta defined as zeros.
        self._prev_density: np.ndarray | None = None

    def reset(self, star: str, seed: int) -> np.ndarray:
        """Reset density cache BEFORE calling super().reset() — super
        calls self._observe() at the end, which would otherwise reuse a
        stale previous-density vector."""
        self._prev_density = None
        return super().reset(star, seed)

    def _observe(self) -> np.ndarray:
        ego = super()._observe()                                     # (7,)
        now = self._compute_density()
        prev = self._prev_density if self._prev_density is not None \
            else np.zeros_like(now)
        delta = now - prev
        self._prev_density = now
        return np.concatenate([ego, now, delta]).astype(np.float32)

    def _compute_density(self) -> np.ndarray:
        """Phase 1: single plane in sim → no neighbors → zeros. Phase 2:
        compute the same hard-binned cubic-falloff density used by the
        watch UI for every aircraft in the sim."""
        if self._sim is None:
            return np.zeros(self.DENSITY_DIM, dtype=np.float32)
        ego = self._sim.aircraft_list.get(self._callsign)
        if ego is None:
            return np.zeros(self.DENSITY_DIM, dtype=np.float32)
        others = [ac for cs, ac in self._sim.aircraft_list.items()
                  if cs != self._callsign]
        if not others:
            return np.zeros(self.DENSITY_DIM, dtype=np.float32)
        return build_density(
            ego, others,
            nm_per_pixel=self._sim.nm_per_pixel,
            cutoff_nm=self._density_cutoff_nm,
        )
