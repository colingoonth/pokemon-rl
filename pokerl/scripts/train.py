"""PPO training entry point.

Single env, small budget, runs locally. This is the "does the loop
actually train?" smoke test. Real runs go to ELSA with vectorized envs
and a multi-hundred-million-step budget.
"""
from __future__ import annotations

from pathlib import Path

import torch

from pokerl.agent.ppo import PPOConfig, train
from pokerl.env.pokemon_red_env import PokemonRedEnv
from pokerl.env.wrappers import FrameStack

ROOT = Path(__file__).resolve().parents[2]


def env_fn():
    return FrameStack(PokemonRedEnv(headless=True, max_steps=2048), k=4)


def main() -> None:
    cfg = PPOConfig(
        total_timesteps=2048,
        n_steps=256,
        n_epochs=4,
        minibatch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"PPO config: {cfg}")
    net = train(env_fn, cfg)
    out = ROOT / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "smoke.pt")
    print(f"Saved checkpoint -> {out / 'smoke.pt'}")


if __name__ == "__main__":
    main()
