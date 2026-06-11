"""Evaluate a trained checkpoint over N deterministic episodes.

Separate from training-time logging — this gives a clean number you can
quote ("the agent earns mean return X over 16 episodes with argmax
policy"). Used post-training and at intermediate checkpoints to track
real progress.

Usage:
  uv run python -m pokerl.eval.rollout --checkpoint checkpoints/smoke.pt
  uv run python -m pokerl.eval.rollout --checkpoint runs/elsa_brock/final.pt --episodes 32
  uv run python -m pokerl.eval.rollout --random --episodes 16
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from pokerl.agent.networks import ActorCritic
from pokerl.env.make import make_env
from pokerl.env.pokemon_red_env import ACTIONS

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to a .pt state dict. Required unless --random.")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=4096,
                   help="Per-episode step budget (matches training default).")
    p.add_argument("--random", action="store_true",
                   help="Use a uniformly-random policy instead of a checkpoint.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from policy instead of argmax.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional CSV path for per-episode results.")
    return p.parse_args()


def load_policy(checkpoint: Path | None, obs_shape: tuple[int, ...], n_actions: int) -> ActorCritic | None:
    if checkpoint is None:
        return None
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    net = ActorCritic(obs_shape, n_actions=n_actions)
    # weights_only=False so full-resume dict checkpoints (with optimizer / numpy
    # normalizer state) load; these are our own trusted files. Full-resume
    # checkpoints wrap weights under "net"; bare/legacy ones are a plain dict.
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded["net"] if isinstance(loaded, dict) and "net" in loaded else loaded
    missing, unexpected = net.load_state_dict(state, strict=False)
    critical = [k for k in missing if k.startswith(("actor.", "backbone.", "fc."))]
    if critical:
        raise SystemExit(
            f"checkpoint is missing policy-critical keys {critical}; "
            f"architecture mismatch — refusing to eval a partly-random net."
        )
    if missing or unexpected:
        print(f"NOTE: load_state_dict missing={sorted(missing)} "
              f"unexpected={sorted(unexpected)}")
    net.eval()
    return net


def main() -> None:
    args = parse_args()
    if args.checkpoint is None and not args.random:
        raise SystemExit("Pass --checkpoint or --random")

    rng = np.random.default_rng(args.seed)

    env = make_env(headless=True, max_steps=args.max_steps, frame_stack=4)
    obs, _ = env.reset(seed=args.seed)

    if args.random:
        net = None
        policy_name = "random"
    else:
        net = load_policy(args.checkpoint, obs.shape, n_actions=len(ACTIONS))
        policy_name = args.checkpoint.name

    returns: list[float] = []
    lengths: list[int] = []
    tile_counts: list[int] = []
    start = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        ep_len = 0
        while True:
            if net is None:
                action = int(rng.integers(0, env.action_space.n))
            else:
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    logits = net(obs_t)[0]  # (logits, value_ext, value_int)
                if args.stochastic:
                    action = int(Categorical(logits=logits).sample().item())
                else:
                    action = int(logits.argmax(dim=-1).item())
            obs, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if term or trunc:
                break
        returns.append(ep_return)
        lengths.append(ep_len)
        tiles = getattr(env.unwrapped.reward_fn, "unique_tiles_visited", 0)  # type: ignore[attr-defined]
        tile_counts.append(int(tiles))

    elapsed = time.time() - start
    env.close()

    arr = np.array(returns)
    tiles = np.array(tile_counts)
    print(f"=== Eval: {policy_name} ({args.episodes} eps, "
          f"{'argmax' if not args.stochastic else 'sample'}) ===")
    print(f"Wall time:            {elapsed:.1f}s")
    print(f"Return  mean / std:   {arr.mean():+.3f} / {arr.std():.3f}")
    print(f"Return  min  / max:   {arr.min():+.3f} / {arr.max():+.3f}")
    print(f"Length  mean:         {np.mean(lengths):.0f}")
    print(f"Tiles   mean / max:   {tiles.mean():.1f} / {tiles.max()}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "return", "length", "unique_tiles_visited"])
            for i, (r, l, t) in enumerate(zip(returns, lengths, tile_counts)):
                w.writerow([i, r, l, t])
        print(f"Wrote per-episode CSV -> {args.out}")


if __name__ == "__main__":
    main()
