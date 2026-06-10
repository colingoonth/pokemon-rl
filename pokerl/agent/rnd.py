"""Random Network Distillation (RND) intrinsic-curiosity module.

RND gives the agent an exploration drive that doesn't depend on hand-placed
rewards: novelty = how badly a trained *predictor* network fails to match a
frozen, randomly-initialized *target* network on the current observation. A
state seen many times is predicted well (low error -> boring); a fresh state
is predicted badly (high error -> "go here"). The predictor catches up as the
agent revisits, so the novelty bonus decays on its own.

Two design choices specific to this project (decided with the audit):

  - The RND nets see a SINGLE grayscale frame (not the 4-stack) plus a small
    PROGRESS VECTOR of hidden story-state bits (parcel obtained, pokedex
    obtained, beat-brock, pokeballs). The frame alone can't distinguish
    "Pallet holding the parcel" from "Pallet before the parcel" — they render
    identically — so curiosity would give the forced parcel backtrack zero
    pull. Feeding the hidden bits makes novelty context-aware. The progress
    vector gets its own encoder branch so a single flag flip meaningfully
    moves the feature (it isn't drowned by the 5,760 frame pixels).

  - Battle frames are excluded from intrinsic reward by the caller (the PPO
    loop zeros it when in_battle != 0): battle RNG is unpredictable noise the
    predictor can never learn (the "noisy-TV" trap), and we don't want the
    agent paid to stare at attack animations.

Normalization (both standard CleanRL-RND details):
  - observations are normalized by a running per-pixel mean/std before the
    nets see them (RND is very sensitive to input scale), then clipped;
  - the intrinsic reward is divided by a running std of the discounted
    intrinsic *returns* so its scale stays stable as the predictor learns.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pokerl.agent.networks import _layer_init


class RunningMeanStd:
    """Welford running mean/variance (numpy). Tracks stats of whatever it's
    fed — observation pixels or intrinsic returns. Parallel batch update."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


class RewardForwardFilter:
    """Maintains the discounted running intrinsic return per env so its std
    can normalize the intrinsic reward. r_t_hat = r_t + gamma * r_{t-1}_hat.
    (CleanRL RND detail — intrinsic reward is non-episodic, never reset.)"""

    def __init__(self, gamma: float, n_envs: int) -> None:
        self.gamma = gamma
        self.rewems = np.zeros(n_envs, dtype=np.float64)

    def update(self, rews: np.ndarray) -> np.ndarray:
        self.rewems = self.rewems * self.gamma + np.asarray(rews, dtype=np.float64)
        return self.rewems


class _RNDEncoder(nn.Module):
    """Maps (single frame, progress vector) -> feature vector. Used identically
    for both the frozen target and the trained predictor."""

    def __init__(
        self,
        frame_shape: tuple[int, int, int],   # (1, H, W)
        progress_dim: int,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = frame_shape
        self.conv = nn.Sequential(
            _layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.frame_fc = _layer_init(nn.Linear(flat, 256))
        # Dedicated branch for the hidden-state bits so a flag flip isn't
        # swamped by the frame; ~20% of the fused feature width.
        self.prog_fc = _layer_init(nn.Linear(progress_dim, 64))
        self.out = _layer_init(nn.Linear(256 + 64, feature_dim))

    def forward(self, frame: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        f = torch.relu(self.frame_fc(self.conv(frame)))
        p = torch.relu(self.prog_fc(progress))
        return self.out(torch.cat([f, p], dim=-1))


class RNDModel(nn.Module):
    """Frozen random target + trained predictor. The per-sample squared error
    between them is the raw novelty signal."""

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        progress_dim: int,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.target = _RNDEncoder(frame_shape, progress_dim, feature_dim)
        self.predictor = _RNDEncoder(frame_shape, progress_dim, feature_dim)
        # Target is fixed forever — it defines the random function the predictor
        # chases. Freeze so DDP / optimizers never touch it.
        for p in self.target.parameters():
            p.requires_grad_(False)

    def novelty(self, frame: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """Per-sample novelty = mean squared (predictor - target) over features.

        The target is always detached. Backprop through the returned tensor
        trains ONLY the predictor (use under torch.enable_grad for the loss;
        under torch.no_grad for the reward). Returns shape (batch,).
        """
        target_feat = self.target(frame, progress).detach()
        pred_feat = self.predictor(frame, progress)
        return (pred_feat - target_feat).pow(2).mean(dim=-1)
