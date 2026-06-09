"""GAE-λ advantage estimation.

Standard generalized-advantage estimation (Schulman et al., 2016):

    δ_t       = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
    A_t       = δ_t + γλ · A_{t+1} · (1 − done_t)
    returns_t = A_t + V(s_t)

The (1 − done) factor stops bootstrapping across episode boundaries — the
"next value" of a terminal step is zero (the reward already accounted for
the outcome).

Operates per-episode on python lists; the trainer stitches the per-episode
arrays into one flat batch afterwards.
"""
from __future__ import annotations

import numpy as np


def compute_gae(rewards: list[float],
                values: list[float],
                dones: list[bool],
                gamma: float,
                lam: float,
                bootstrap_value: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """One trajectory in, (advantages, returns) out — both shape (T,).

    `bootstrap_value` is V(s_T) for an episode that was truncated without
    a terminal flag (not used here since our episodes always terminate at
    success/timeout/exit, all of which set done=True). Provided so the
    function is general.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    next_value = bootstrap_value
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns
