"""PPO policy: BC GMM actor + value network, with the act / evaluate
interface PPO needs.

Loads the BC checkpoint into a fresh `BCActor` (so the model is trainable
and not wrapped in the BC `Runtime` inference glue), and pairs it with a
fresh `ValueNetwork`. All standardization buffers (target_mean/target_std
on the actor) restore via state_dict.

Three methods used by the trainer:

  act(obs)             — stochastic action + log_prob + value, no grad
  evaluate_actions     — log_prob + entropy + value w.r.t. *current* policy
                         for the actions stored from rollouts
  load_state           — load a state_dict (used by mp workers to refresh
                         their local copy each PPO iteration)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from rl_bc.bc_gmm.model import BCActor, build_from_saved
from .critic import ValueNetwork


class PPOPolicy(nn.Module):
    def __init__(self, actor: BCActor, critic: ValueNetwork):
        super().__init__()
        self.actor = actor
        self.critic = critic

    # ---------------------------------------------------------- #
    # Constructors / serialization
    # ---------------------------------------------------------- #

    @classmethod
    def from_bc_checkpoint(cls, ckpt_path: str | Path,
                           value_hidden: int = 64,
                           value_dropout: float = 0.0,
                           device: str | torch.device = 'cpu'
                           ) -> 'PPOPolicy':
        device = torch.device(device)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor = build_from_saved(blob['config']).to(device)
        actor.load_state_dict(blob['model_state'])
        # PPO's env already passes the observation as a vector sliced to
        # whatever columns the BC actor was trained on (see
        # `rl_ppo/env.py::_observe`). To avoid the actor re-slicing that
        # vector with its own input_indices, set the actor's index list
        # to identity over its encoder's `in_features`.
        in_features = actor.encoder[0].in_features
        actor.input_indices = list(range(in_features))
        actor._idx_t = None
        critic = ValueNetwork(input_dim=in_features, hidden=value_hidden,
                              dropout=value_dropout).to(device)
        return cls(actor, critic)

    def state_dict_split(self) -> dict:
        """Picklable dict for cross-process state sync (workers reload this
        before each rollout iteration so they always sample from the latest
        policy)."""
        return {
            'actor': {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            'critic': {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
        }

    def load_state_split(self, state: dict, device: str | torch.device = 'cpu'):
        device = torch.device(device)
        self.actor.load_state_dict({k: v.to(device) for k, v in state['actor'].items()})
        self.critic.load_state_dict({k: v.to(device) for k, v in state['critic'].items()})

    # ---------------------------------------------------------- #
    # Acting (rollout side)
    # ---------------------------------------------------------- #

    @torch.no_grad()
    def act(self, obs: torch.Tensor,
            generator: torch.Generator | None = None) -> dict:
        """Sample one stochastic action per observation row.

        `generator` is forwarded to the GMM's component-categorical draw
        AND to the vMF / Gaussian conditionals — matches BC Runtime's
        per-aircraft seeded behavior when passed; uses the global RNG
        when None.

        Returns a dict with:
          action     (B, 4)  — physical units (sin θ, cos θ, alt_kft, spd_norm)
          log_prob   (B,)    — log π(action | obs) under current policy
                                (NB: log_prob does NOT depend on `generator`
                                 — it's the density at the sampled point,
                                 unaffected by how the sample was drawn)
          value      (B,)    — V(obs)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        c = self.actor.encode(obs)
        sample = self.actor.sample(c, generator=generator, deterministic=False)
        log_prob = self.actor.log_prob(sample, c)
        value = self.critic(obs)
        return {'action': sample, 'log_prob': log_prob, 'value': value}

    # ---------------------------------------------------------- #
    # Evaluating (PPO-update side, needs grad)
    # ---------------------------------------------------------- #

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor) -> dict:
        """Re-score `actions` under the *current* policy.

        Returns a dict with:
          log_prob   (B,)  — log π_θ(a | s)
          entropy    (B,)  — sample-based estimate: -log_prob  (Jensen-ish);
                              proper mixture entropy is intractable
          value      (B,)  — V_θ(s)
        """
        c = self.actor.encode(obs)
        log_prob = self.actor.log_prob(actions, c)
        # Single-sample MC estimate of entropy of π(·|s); biased but the
        # standard choice when closed-form entropy isn't available (e.g.
        # mixture or normalizing-flow policies).
        entropy = -log_prob
        value = self.critic(obs)
        return {'log_prob': log_prob, 'entropy': entropy, 'value': value}
