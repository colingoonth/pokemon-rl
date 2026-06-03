"""Actor-critic network for PPO on PokemonRedEnv.

Shared NatureCNN backbone (3 conv layers + FC 512) feeds two heads:
  - actor: logits over the discrete action space
  - critic: scalar value estimate V(s)

Input is (batch, frame_stack, H, W) uint8 — scaled to [0, 1] inside
forward(). Output is (logits, value) for downstream PPO use.

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
        # policy), gain=1 on critic.
        self.actor = _layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is uint8 in [0, 255]; PPO expects float in [0, 1]
        x = obs.float() / 255.0
        x = self.backbone(x)
        return self.fc(x)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs)).squeeze(-1)
