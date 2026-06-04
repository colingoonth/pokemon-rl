"""Press a single direction repeatedly; log position every step.

Hypothesis under test: the env's button-press handling doesn't hold long
enough for Pokemon Red's movement code to register, so even deterministic
"keep pressing up" doesn't actually move the player. If that's true, no
policy could possibly learn to explore regardless of reward design.

Output: per-step (map_id, x, y) and whether it changed since the last
step. Final summary tells us how many distinct positions the player
visited.
"""
from __future__ import annotations

import argparse

from pokerl.env import ram_map as rm
from pokerl.env.pokemon_red_env import ACTIONS, PokemonRedEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--action", default="up", choices=list(ACTIONS),
                   help="Action to press every step.")
    p.add_argument("--steps", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    action_idx = ACTIONS.index(args.action)

    env = PokemonRedEnv(headless=True, max_steps=args.steps + 1)
    env.reset(seed=0)

    mem = env.pyboy.memory
    positions: list[tuple[int, int, int]] = []
    start_pos = (rm.map_id(mem), *rm.player_position(mem))
    positions.append(start_pos)
    print(f"start: map=0x{start_pos[0]:02X} pos={start_pos[1:]}")

    for step in range(args.steps):
        env.step(action_idx)
        mem = env.pyboy.memory
        pos = (rm.map_id(mem), *rm.player_position(mem))
        changed = "  " if pos == positions[-1] else "->"
        print(f"step {step + 1:3d}  {changed}  map=0x{pos[0]:02X} pos={pos[1:]}")
        positions.append(pos)

    env.close()

    distinct = set(positions)
    print()
    print(f"Pressed '{args.action}' {args.steps} times.")
    print(f"Distinct (map, x, y) tuples visited: {len(distinct)}")
    if len(distinct) == 1:
        print(
            "\nFAIL: Position never changed across all steps.\n"
            "Either:\n"
            "  - The button press duration is too short for Pokemon Red\n"
            "    to register a movement command\n"
            "  - The player is against a wall in the spawn direction\n"
            "  - The RAM addresses for position aren't updating live"
        )
    else:
        print("\nPASS: position changes — env input handling is fine.")


if __name__ == "__main__":
    main()
