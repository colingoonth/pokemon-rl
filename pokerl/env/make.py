"""Single source of truth for env construction.

Both train and eval scripts call make_env() so the wrapping is identical
across the two paths. Diverging here is a very common silent bug in RL
projects (the agent sees a different obs at eval time than at train time
and acts confused).
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from pokerl.env.pokemon_red_env import PokemonRedEnv
from pokerl.env.rewards import Reward
from pokerl.env.wrappers import FrameStack


def make_env(
    rom_path: str | Path | None = None,
    state_path: str | Path | None = None,
    headless: bool = True,
    max_steps: int = 4096,
    frame_stack: int = 4,
    reward: Reward | None = None,
) -> gym.Env:
    env_kwargs: dict = {"headless": headless, "max_steps": max_steps, "reward": reward}
    if rom_path is not None:
        env_kwargs["rom_path"] = rom_path
    if state_path is not None:
        env_kwargs["state_path"] = state_path
    env = PokemonRedEnv(**env_kwargs)
    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)
    return env
