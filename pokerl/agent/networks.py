"""Actor-critic network for PPO on PokemonRedEnv.

Shared NatureCNN backbone (3 conv layers + FC 512) feeds three heads:
  - actor:      logits over the discrete action space
  - critic:     extrinsic state value V_ext(s)
  - critic_int: intrinsic (RND-novelty) state value V_int(s)

The intrinsic head exists so the RND curiosity build can value the
(non-episodic, separately-discounted) intrinsic-reward stream independently
of the extrinsic stream — the standard two-value-head RND design. When RND
is disabled the intrinsic head is simply unused (its output ignored), so
this network is a drop-in for the plain-PPO path too.

Input is (batch, frame_stack, H, W) uint8 — scaled to [0, 1] inside
forward(). Output is (logits, value_ext, value_int).

The backbone/fc/actor/critic parameter names are unchanged from the
two-head version so a pre-RND checkpoint warm-starts cleanly: only
critic_int is a new key (loaded fresh via strict=False).

Orthogonal init with sqrt(2) gain on hidden layers and small gain
(0.01 actor / 1.0 critic) on output heads. These are the well-known
PPO implementation details from CleanRL's ppo_atari.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _layer_init(layer: nn.Module, std: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)  # type: ignore[arg-type]
    nn.init.constant_(layer.bias, bias)     # type: ignore[arg-type]
    return layer


class ActorCritic(nn.Module):
    """NatureCNN backbone + policy and value heads."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],   # (C, H, W) — C is frame_stack
        n_actions: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        c, h, w = obs_shape

        self.backbone = nn.Sequential(
            _layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Probe conv output size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.backbone(dummy).shape[1]

        self.fc = nn.Sequential(
            _layer_init(nn.Linear(flat_dim, hidden_dim)),
            nn.ReLU(),
        )

        # Output heads: small init on actor (encourages near-uniform initial
        # policy), gain=1 on critics.
        self.actor = _layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        # Intrinsic-value head (RND). New key vs the pre-RND checkpoint, so it
        # initializes fresh on a strict=False warm-start.
        self.critic_int = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is uint8 in [0, 255]; PPO expects float in [0, 1]
        x = obs.float() / 255.0
        x = self.backbone(x)
        return self.fc(x)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        return (
            self.actor(h),
            self.critic(h).squeeze(-1),
            self.critic_int(h).squeeze(-1),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Extrinsic value V_ext(s)."""
        return self.critic(self.features(obs)).squeeze(-1)

    def value_int(self, obs: torch.Tensor) -> torch.Tensor:
        """Intrinsic value V_int(s) (RND novelty stream)."""
        return self.critic_int(self.features(obs)).squeeze(-1)
