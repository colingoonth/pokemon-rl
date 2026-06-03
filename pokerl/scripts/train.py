"""PPO training entry point.

Single env, small budget, runs locally. This is the "does the loop
actually train?" smoke test. Real runs go to ELSA with vectorized envs
and a multi-hundred-million-step budget.
"""
from __future__ import annotations

from pathlib import Path

import torch

from pokerl.agent.ppo import PPOConfig, train
from pokerl.env.make import make_vec_env

ROOT = Path(__file__).resolve().parents[2]

N_ENVS = 4   # local default; ELSA config will override


def env_fn():
    return make_vec_env(n_envs=N_ENVS, headless=True, max_steps=2048, frame_stack=4)


def main() -> None:
    out = ROOT / "runs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig(
        total_timesteps=4096,
        n_envs=N_ENVS,
        n_steps=128,
        n_epochs=4,
        minibatch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_csv=str(out / "metrics.csv"),
    )
    print(f"PPO config: {cfg}")
    net = train(env_fn, cfg)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt_dir / "smoke.pt")
    print(f"Saved checkpoint -> {ckpt_dir / 'smoke.pt'}")
    print(f"Metrics CSV     -> {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
