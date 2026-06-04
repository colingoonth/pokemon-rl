"""Single source of truth for env construction.

Both train and eval scripts call make_env() so the wrapping is identical
across the two paths. Diverging here is a very common silent bug in RL
projects (the agent sees a different obs at eval time than at train time
and acts confused).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

from pokerl.env.pokemon_red_env import PokemonRedEnv
from pokerl.env.rewards import Reward, RewardV0_1
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


def make_vec_env(
    n_envs: int,
    rom_path: str | Path | None = None,
    state_path: str | Path | None = None,
    headless: bool = True,
    max_steps: int = 4096,
    frame_stack: int = 4,
    reward_cls: Callable[[], Reward] = RewardV0_1,
    async_envs: bool = False,
) -> VectorEnv:
    """Construct a vectorized env with `n_envs` parallel PokemonRed instances.

    Each env gets its own freshly-constructed Reward instance so per-env
    exploration sets don't bleed across envs.

    async_envs=True puts each env in its own subprocess (gym.vector.
    AsyncVectorEnv). True parallelism — sidesteps the Python GIL so 32
    PyBoy instances actually run in parallel instead of serially.
    Recommended on real training runs. Sync is fine for unit tests and
    small dev loops.
    """

    def _make_one() -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            return make_env(
                rom_path=rom_path,
                state_path=state_path,
                headless=headless,
                max_steps=max_steps,
                frame_stack=frame_stack,
                reward=reward_cls(),
            )
        return thunk

    env_fns = [_make_one() for _ in range(n_envs)]
    if async_envs:
        # context="spawn": each env subprocess gets a fresh Python interpreter.
        # Required when the parent has an initialized CUDA context (DDP ranks),
        # because fork-after-CUDA-init raises "Cannot re-initialize CUDA in
        # forked subprocess." Safe in single-process mode too.
        return AsyncVectorEnv(env_fns, context="spawn")
    return SyncVectorEnv(env_fns)
