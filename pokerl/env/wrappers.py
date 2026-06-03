"""Gymnasium wrappers for PokemonRedEnv."""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStack(gym.ObservationWrapper):
    """Stack the last `k` frames as channels.

    Input observation shape:  (H, W) uint8
    Output observation shape: (k, H, W) uint8

    The first observation after reset is the spawn frame repeated k times.
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = k
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 2, f"FrameStack expects 2D obs, got {obs_shape}"
        h, w = obs_shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, h, w), dtype=np.uint8
        )
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(obs)
        return np.stack(self._frames, axis=0), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        return np.stack(self._frames, axis=0)
