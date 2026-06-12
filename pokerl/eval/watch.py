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
from pokerl.env.rewards import get_reward_cls

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "checkpoints" / "smoke.pt"
OUT_DIR = ROOT / "notes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--record", action="store_true",
                   help="Run headless and dump a gif instead of opening a window")
    p.add_argument("--silent", action="store_true",
                   help="Run headless with no window and no gif — just print "
                        "the metrics + reward attribution at the end.")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax actions instead of sampling")
    p.add_argument("--speed", type=float, default=0.0,
                   help="Emulation speed multiplier in SDL2 mode "
                        "(1.0=real time, 5.0=5x, 0=unbounded — default). "
                        "Ignored when --record.")
    p.add_argument("--state-path", type=Path, default=None,
                   help="Override env start state. None = env default "
                        "(post_intro.state). Use this to eval a policy "
                        "from the same state it was trained on.")
    p.add_argument("--reward-class", type=str, default="RewardV0_4_2_center",
                   help="Reward class name. Must match the class the "
                        "checkpoint was trained against — otherwise "
                        "attribution and totals won't match training.")
    return p.parse_args()


def load_net(checkpoint: Path, obs_shape: tuple[int, ...], n_actions: int) -> ActorCritic:
    net = ActorCritic(obs_shape, n_actions=n_actions)
    if checkpoint.exists():
        # weights_only=False so full-resume dict checkpoints (with optimizer /
        # numpy normalizer state) load; these are our own trusted files.
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Full-resume checkpoints wrap weights under "net"; bare/legacy and
        # weights-only warm-start checkpoints are a plain state_dict.
        state = loaded["net"] if isinstance(loaded, dict) and "net" in loaded else loaded
        # strict=False so the watcher loads both old 2-head checkpoints (no
        # critic_int) and new 3-head RND checkpoints. But DON'T swallow the
        # result: a missing actor/backbone key means we'd silently eval a
        # partly-random net, which looks like a "trained" policy behaving badly.
        missing, unexpected = net.load_state_dict(state, strict=False)
        critical = [k for k in missing if k.startswith(("actor.", "backbone.", "fc."))]
        if critical:
            raise RuntimeError(
                f"checkpoint is missing policy-critical keys {critical}; "
                f"architecture mismatch — refusing to eval a partly-random net."
            )
        if missing or unexpected:
            print(f"NOTE: load_state_dict missing={sorted(missing)} "
                  f"unexpected={sorted(unexpected)}")
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"NOTE: no checkpoint at {checkpoint}; running with untrained policy")
    net.eval()
    return net


def main() -> None:
    args = parse_args()

    headless = args.record or args.silent
    reward_cls = get_reward_cls(args.reward_class)
    env = make_env(headless=headless, max_steps=args.steps + 1,
                   frame_stack=4, state_path=args.state_path,
                   reward=reward_cls())
    print(f"Reward: {args.reward_class}")
    if not headless:
        # Throttle the SDL2 window to the requested multiple of real time.
        env.unwrapped.pyboy.set_emulation_speed(args.speed)
    obs, _ = env.reset()
    net = load_net(args.checkpoint, obs.shape, n_actions=len(ACTIONS))

    frames: list[np.ndarray] = []
    if args.record:
        frames.append(env.render())

    total_reward = 0.0
    tile_count_start = env.unwrapped.reward_fn.unique_tiles_visited  # type: ignore[attr-defined]
    # Reward attribution aggregation.
    # Components: cumulative sum per named component.
    # Fires: count of steps that contributed > 0 (negative or positive).
    comp_total: dict[str, float] = {}
    comp_fires: dict[str, int] = {}
    # First step at which each story rung (RUNG_*) fired this rollout.
    rung_step: dict[str, int] = {}
    start = time.time()

    for step in range(args.steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits = net(obs_t)[0]  # (logits, value_ext, value_int) — only logits needed
        if args.deterministic:
            action = int(logits.argmax(dim=-1).item())
        else:
            action = int(Categorical(logits=logits).sample().item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        # Aggregate reward attribution. Reward classes that support the
        # last_components dict (V0.4.0+) populate it on each compute().
        rf = env.unwrapped.reward_fn  # type: ignore[attr-defined]
        components = getattr(rf, "last_components", None)
        if components:
            for name, val in components.items():
                comp_total[name] = comp_total.get(name, 0.0) + val
                comp_fires[name] = comp_fires.get(name, 0) + 1
                if name.startswith("RUNG_") and name not in rung_step:
                    rung_step[name] = step + 1

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

    if comp_total:
        print()
        print("=== Reward attribution ===")
        # Sort by absolute magnitude so the loudest components surface first.
        ordered = sorted(comp_total.items(), key=lambda kv: -abs(kv[1]))
        name_w = max(len(n) for n, _ in ordered)
        attrib_sum = 0.0
        for name, total in ordered:
            fires = comp_fires.get(name, 0)
            attrib_sum += total
            share = (total / total_reward * 100) if total_reward != 0 else 0.0
            print(f"  {name:<{name_w}}  {total:+10.3f}  "
                  f"({fires:>5} fires, {share:+6.1f}% of total)")
        print(f"  {'sum':<{name_w}}  {attrib_sum:+10.3f}")
        residual = total_reward - attrib_sum
        if abs(residual) > 1e-4:
            print(f"  {'unattributed':<{name_w}}  {residual:+10.3f}")

    # Story-ladder detail (V0.5.1+ storyladder reward classes expose _LADDER /
    # _mult / _fired). Shows which rungs were reached, in path order, the step
    # each fired, and the running story multiplier — so a single silent run
    # reads as "how far up the story did it climb."
    ladder = getattr(rf, "_LADDER", None)
    if ladder is not None:
        fired = getattr(rf, "_fired", set())
        final_mult = getattr(rf, "_mult", 1.0)
        reached = [e[0] for e in ladder if e[0] in fired]
        furthest = reached[-1] if reached else "(none)"
        print()
        print("=== Story ladder ===")
        print(f"  multiplier reached:  {final_mult:.2f}x")
        print(f"  rungs reached:       {len(reached)}/{len(ladder)}   furthest: {furthest}")
        key_w = max(len(e[0]) for e in ladder)
        m = 1.0
        for key, _kind, _target, bonus, inc in ladder:
            if key in fired:
                m += inc
                rk = "RUNG_" + key
                when = f"step {rung_step[rk]}" if rk in rung_step else "pre-masked at reset"
                print(f"  [x] {key:<{key_w}}  one-shot +{bonus:>5.1f}   M={m:4.2f}   {when}")
            else:
                print(f"  [ ] {key:<{key_w}}  (would add +{inc:.2f} mult)")

    if args.record:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "watch_rollout.gif"
        iio.imwrite(out, np.array(frames), duration=80, loop=0)
        print(f"Wrote gif -> {out}")


if __name__ == "__main__":
    main()
