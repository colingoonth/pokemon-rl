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

import math

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


class _RNDStateEncoder(nn.Module):
    """Maps (RAM state [map_id, x, y], progress vector) -> feature vector.

    The animation-invariant alternative to _RNDEncoder, for the "ram" RND mode.
    Frame-RND paid the agent for raw pixel novelty, so on-cycle overworld
    animation (water, flowers) kept emitting novelty for a *stationary* agent —
    a noisy-TV loiter trap — and tiles that render identically before/after a
    story flag gave forced backtracks zero pull. Keying novelty on RAM coords
    removes the animation entirely (the same tile is the same input whether or
    not the water is shimmering), and fusing the hidden progress bits makes a
    single story-flag flip refresh novelty across the whole already-explored
    map — which is what drives the parcel -> pokedex -> pokeballs backtrack.

      - map_id -> learnable embedding: sharp, per-map identity.
      - (x, y) -> FIXED random Fourier features: high-frequency so an UNVISITED
        tile reads as genuinely novel instead of bleeding into visited
        neighbours (the over-smoothing failure of feeding raw coord floats).
        `fourier_scale` is the spatial-granularity knob: higher = sharper
        per-tile resolution, lower = smoother coverage.
      - progress -> dedicated branch (~20% of fused width), same rationale as
        the frame encoder: a lone flag flip must move the feature, not drown.
    """

    def __init__(
        self,
        progress_dim: int,
        num_maps: int = 256,
        map_embed_dim: int = 32,
        n_fourier: int = 128,
        fourier_scale: float = 16.0,
        coord_scale: float = 256.0,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_maps = num_maps
        self.coord_scale = float(coord_scale)
        self.map_embed = nn.Embedding(num_maps, map_embed_dim)
        # Fixed random Fourier basis for (x, y). A buffer (not a Parameter) so it
        # is frozen in BOTH target and predictor, never trained, and persists
        # across resume. Each model instance draws its own basis — fine for RND,
        # since the predictor only needs to approximate the target as a function
        # of the raw state, not share its featurisation.
        self.register_buffer("fourier_B", torch.randn(2, n_fourier) * fourier_scale)
        spatial_in = map_embed_dim + 2 * n_fourier
        self.spatial_fc = _layer_init(nn.Linear(spatial_in, 256))
        self.prog_fc = _layer_init(nn.Linear(progress_dim, 64))
        self.out = _layer_init(nn.Linear(256 + 64, feature_dim))

    def forward(self, state: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        # state: (batch, 3) = [map_id, x, y] as floats.
        map_id = state[:, 0].long().clamp_(0, self.num_maps - 1)
        xy = state[:, 1:3] / self.coord_scale                  # ~[0, 1]
        proj = 2.0 * math.pi * (xy @ self.fourier_B)           # (batch, n_fourier)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        m = self.map_embed(map_id)
        s = torch.relu(self.spatial_fc(torch.cat([m, ff], dim=-1)))
        p = torch.relu(self.prog_fc(progress))
        return self.out(torch.cat([s, p], dim=-1))


class RNDModel(nn.Module):
    """Frozen random target + trained predictor. The per-sample squared error
    between them is the raw novelty signal.

    `input_mode` selects the encoder: "frame" (the original conv-over-pixels +
    progress) or "ram" (animation-invariant map_id/x/y + progress). The frame
    args are ignored in ram mode and vice-versa.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        progress_dim: int,
        feature_dim: int = 256,
        input_mode: str = "frame",
        num_maps: int = 256,
        n_fourier: int = 128,
        fourier_scale: float = 16.0,
    ) -> None:
        super().__init__()
        self.input_mode = input_mode
        if input_mode == "ram":
            def mk() -> nn.Module:
                return _RNDStateEncoder(
                    progress_dim, num_maps=num_maps, n_fourier=n_fourier,
                    fourier_scale=fourier_scale, feature_dim=feature_dim,
                )
        elif input_mode == "frame":
            def mk() -> nn.Module:
                return _RNDEncoder(frame_shape, progress_dim, feature_dim)
        else:
            raise ValueError(f"unknown RND input_mode {input_mode!r} (expected 'frame' or 'ram')")
        self.target = mk()
        self.predictor = mk()
        # Target is fixed forever — it defines the random function the predictor
        # chases. Freeze so DDP / optimizers never touch it.
        for p in self.target.parameters():
            p.requires_grad_(False)

    def novelty(self, obs: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """Per-sample novelty = mean squared (predictor - target) over features.

        `obs` is the normalized frame (frame mode) or the raw [map_id, x, y]
        state vector (ram mode). The target is always detached. Backprop through
        the returned tensor trains ONLY the predictor (use under
        torch.enable_grad for the loss; under torch.no_grad for the reward).
        Returns shape (batch,).
        """
        target_feat = self.target(obs, progress).detach()
        pred_feat = self.predictor(obs, progress)
        return (pred_feat - target_feat).pow(2).mean(dim=-1)
