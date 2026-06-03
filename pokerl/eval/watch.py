"""Load a checkpoint and watch the trained agent play.

Two modes:
  - --record: headless run, dump a gif of the rollout
  - default:  SDL2 window, watch live at game speed

Usage:
  uv run python -m pokerl.eval.watch
  uv run python -m pokerl.eval.watch --checkpoint checkpoints/smoke.pt --steps 1000
  uv run python -m pokerl.eval.watch --record --steps 600
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from torch.distributions import Categorical

from pokerl.agent.networks import ActorCritic
from pokerl.env.make import make_env
from pokerl.env.pokemon_red_env import ACTIONS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "checkpoints" / "smoke.pt"
OUT_DIR = ROOT / "notes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--record", action="store_true",
                   help="Run headless and dump a gif instead of opening a window")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax actions instead of sampling")
    return p.parse_args()


def load_net(checkpoint: Path, obs_shape: tuple[int, ...], n_actions: int) -> ActorCritic:
    net = ActorCritic(obs_shape, n_actions=n_actions)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"NOTE: no checkpoint at {checkpoint}; running with untrained policy")
    net.eval()
    return net


def main() -> None:
    args = parse_args()

    env = make_env(headless=args.record, max_steps=args.steps + 1, frame_stack=4)
    obs, _ = env.reset()
    net = load_net(args.checkpoint, obs.shape, n_actions=len(ACTIONS))

    frames: list[np.ndarray] = []
    if args.record:
        frames.append(env.render())

    total_reward = 0.0
    tile_count_start = env.unwrapped.reward_fn.unique_tiles_visited  # type: ignore[attr-defined]
    start = time.time()

    for step in range(args.steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(obs_t)
        if args.deterministic:
            action = int(logits.argmax(dim=-1).item())
        else:
            action = int(Categorical(logits=logits).sample().item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if args.record and step % 4 == 0:
            frames.append(env.render())

        if terminated or truncated:
            break

    elapsed = time.time() - start
    tile_count_end = env.unwrapped.reward_fn.unique_tiles_visited  # type: ignore[attr-defined]
    env.close()

    print(f"Rolled {step + 1} steps in {elapsed:.2f}s "
          f"({(step + 1) / elapsed:.1f} steps/s)")
    print(f"Total return:            {total_reward:+.3f}")
    print(f"Unique tiles visited:    {tile_count_end - tile_count_start + 1}")
    if args.record:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "watch_rollout.gif"
        iio.imwrite(out, np.array(frames), duration=80, loop=0)
        print(f"Wrote gif -> {out}")


if __name__ == "__main__":
    main()
