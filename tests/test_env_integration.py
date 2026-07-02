"""End-to-end: env -> FrameStack -> ActorCritic forward pass.

Skips if the ROM or save state isn't present. Verifies that the actual
observation produced by the env can be consumed by the network without
shape errors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ROM = ROOT / "roms" / "pokemon_red.gb"
STATE = ROOT / "states" / "post_intro.state"

pytestmark = pytest.mark.skipif(
    not STATE.exists() or not ROM.exists(),
    reason="ROM or save state missing",
)


def test_env_stacked_obs_through_network():
    from pokerl.agent.networks import ActorCritic
    from pokerl.env.pokemon_red_env import ACTIONS, PokemonRedEnv
    from pokerl.env.ram_map import PROGRESS_DIM
    from pokerl.env.wrappers import FrameStack

    env = FrameStack(PokemonRedEnv(headless=True, max_steps=10), k=4)
    try:
        obs, _ = env.reset()
        assert obs.shape == (4, 72, 80)
        assert obs.dtype == np.uint8

        net = ActorCritic(obs.shape, n_actions=len(ACTIONS))
        prog_t = torch.zeros(1, PROGRESS_DIM, dtype=torch.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            logits, value, value_int = net(obs_t, prog_t)
        assert logits.shape == (1, len(ACTIONS))
        assert value.shape == (1,)
        assert value_int.shape == (1,)

        # Run one env step, network should still consume the result
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert obs.shape == (4, 72, 80)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            net(obs_t, prog_t)
    finally:
        env.close()
