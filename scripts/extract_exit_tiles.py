"""Empirically derive per-map EXIT tiles for the V0.5.8 within-map field
refinement. Runs the 100M warm-start policy (which reaches Viridian, so it
crosses every forward-leg boundary) and records, for each map transition
FROM->TO, the (x,y) on the FROM map at the last step before the crossing. The
modal (x,y) per edge is that map's exit toward that neighbor.

Local-first, no cluster. Usage:
    uv run python -m scripts.extract_exit_tiles --checkpoint checkpoints/cmp/v5_1_warmstart_final.pt --steps 60000
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.distributions import Categorical

from pokerl.env import ram_map as rm
from pokerl.env.make import make_env
from pokerl.env.pokemon_red_env import ACTIONS
from pokerl.env.rewards import get_reward_cls
from pokerl.eval.watch import load_net

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "checkpoints/cmp/v5_1_warmstart_final.pt")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--state-path", type=str, default="states/blue_fight.state")
    args = p.parse_args()

    env = make_env(headless=True, max_steps=args.steps + 1, frame_stack=4,
                   state_path=args.state_path,
                   reward=get_reward_cls("RewardV0_5_5_storyladder")())
    obs, _ = env.reset()
    net = load_net(args.checkpoint, obs.shape, n_actions=len(ACTIONS))
    mem = env.unwrapped.pyboy.memory  # type: ignore[attr-defined]

    # edge (from_map, to_map) -> Counter of (x,y) on from_map just before crossing
    edges: dict[tuple[int, int], Counter] = defaultdict(Counter)
    transitions: Counter = Counter()  # (from,to) -> count

    prev_map = rm.map_id(mem)
    prev_xy = rm.player_position(mem)
    for _ in range(args.steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits = net(obs_t)[0]
        action = int(Categorical(logits=logits).sample().item())
        obs, _, term, trunc, _ = env.step(action)
        cur_map = rm.map_id(mem)
        cur_xy = rm.player_position(mem)
        if cur_map != prev_map:
            edges[(prev_map, cur_map)][prev_xy] += 1
            transitions[(prev_map, cur_map)] += 1
        prev_map, prev_xy = cur_map, cur_xy
        if term or trunc:
            obs, _ = env.reset()
            prev_map = rm.map_id(mem)
            prev_xy = rm.player_position(mem)
    env.close()

    name = {v: k for k, v in vars(rm).items() if k.startswith("MAP_")}
    print(f"\n=== transitions observed ({sum(transitions.values())} total) ===")
    for (fm, to), n in sorted(transitions.items(), key=lambda kv: -kv[1]):
        fn = name.get(fm, hex(fm)); tn = name.get(to, hex(to))
        modal_xy, modal_n = edges[(fm, to)].most_common(1)[0]
        print(f"  {fn}({hex(fm)}) -> {tn}({hex(to)}):  n={n:4d}  "
              f"exit(x,y)={modal_xy}  (modal {modal_n}/{n})")

    print("\n=== EXIT_TILE suggestion (from_map, to_map) -> (x, y) ===")
    print("_EXIT_TILE = {")
    for (fm, to), c in sorted(edges.items()):
        if transitions[(fm, to)] >= 3:  # only edges seen enough to trust
            modal_xy = c.most_common(1)[0][0]
            print(f"    ({hex(fm)}, {hex(to)}): {modal_xy},")
    print("}")


if __name__ == "__main__":
    main()
