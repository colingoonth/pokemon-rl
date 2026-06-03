"""PPO training entry point.

Usage:
  uv run python -m pokerl.scripts.train --config configs/dev_local.yaml
  uv run python -m pokerl.scripts.train --config configs/elsa_brock.yaml

Run artifacts (csv log, checkpoint) land in runs/<run_name>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pokerl.agent.ppo import train
from pokerl.env.make import make_vec_env
from pokerl.env.rewards import get_reward_cls
from pokerl.infra.config import load_ppo_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "dev_local.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--device", default=None,
                   help="Override device (cpu / cuda). Default: auto.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Warm-start: load this checkpoint into the policy "
                        "before training begins. Useful when continuing an "
                        "earlier run; harmful when the earlier run learned "
                        "a degenerate policy you want to discard.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_root = ROOT / "runs"
    cfg, run_name = load_ppo_config(args.config, device=device)
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_csv = str(run_dir / "metrics.csv")

    print(f"Config: {args.config}")
    print(f"Run:    {run_name}")
    print(f"Device: {device}")
    print(f"PPO:    {cfg}")

    reward_cls = get_reward_cls(cfg.reward_class)
    print(f"Reward: {cfg.reward_class}")

    def env_fn():
        return make_vec_env(
            n_envs=cfg.n_envs,
            headless=True,
            max_steps=4096,
            frame_stack=4,
            async_envs=cfg.async_envs,
            reward_cls=reward_cls,
        )

    net = train(env_fn, cfg, resume_path=args.resume)
    torch.save(net.state_dict(), run_dir / "final.pt")
    print(f"Saved final checkpoint -> {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
