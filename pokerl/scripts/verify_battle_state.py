"""Load a save state and dump every battle-related RAM read.

Use to confirm the new RAM addresses in ram_map are correct before
trusting them in a reward function. Run with a state file captured
INSIDE a battle (wild or trainer) via:

  uv run python -m pokerl.scripts.make_save_state --out states/wild_battle.state
  # play into a wild battle, close window
  uv run python -m pokerl.scripts.verify_battle_state states/wild_battle.state

The script prints what each address reads. Sanity-check against what
you see on screen — if the enemy is a Pidgey, species should be the
internal id for Pidgey (0x24 in Gen 1's internal table).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pyboy import PyBoy

from pokerl.env import ram_map as rm

ROOT = Path(__file__).resolve().parents[2]
ROM = ROOT / "roms" / "pokemon_red.gb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("state", type=Path, help="Path to the save state to inspect.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.state.exists():
        raise SystemExit(f"State not found: {args.state}")

    pyboy = PyBoy(str(ROM), window="null")
    try:
        with args.state.open("rb") as f:
            pyboy.load_state(f)
        pyboy.tick()
        mem = pyboy.memory

        battle_state = rm.in_battle(mem)
        battle_label = {0: "overworld", 1: "wild", 2: "trainer"}.get(battle_state, "?")

        print("=== Battle-state RAM readings ===")
        print()
        print(f"in_battle:           {battle_state} ({battle_label})")
        print()
        print("--- Enemy Pokemon (read meaningfully only when in_battle != 0) ---")
        print(f"  enemy species:     0x{rm.enemy_mon_species(mem):02X}  ({rm.enemy_mon_species(mem)})")
        print(f"  enemy HP:          {rm.enemy_mon_hp(mem)}")
        print()
        print("--- Trainer (read meaningfully only when in_battle == 2) ---")
        cls, num = rm.trainer_id(mem)
        print(f"  trainer class:     0x{cls:02X}  ({cls})")
        print(f"  trainer number:    0x{num:02X}  ({num})")
        print()
        print("--- Player party context ---")
        print(f"  party count:       {rm.party_count(mem)}")
        print(f"  party species:     {[hex(s) for s in rm.party_species(mem)]}")
        print(f"  party levels:      {rm.party_levels(mem)}")
        print(f"  party HP:          {rm.party_hp(mem)}")
        print()
        print("--- Location ---")
        print(f"  map_id:            0x{rm.map_id(mem):02X}")
        print(f"  player position:   {rm.player_position(mem)}")
        print()
        if battle_state == 0:
            print(
                "NOTE: in_battle = 0. The enemy species / HP / trainer reads above\n"
                "are stale values from whatever battle was last active (or zeros).\n"
                "To verify those addresses, capture a state DURING a battle:\n"
                "  uv run python -m pokerl.scripts.make_save_state --out states/wild_battle.state\n"
                "  # walk into Route 1 grass, trigger a wild battle, close the window mid-battle\n"
                "  uv run python -m pokerl.scripts.verify_battle_state states/wild_battle.state"
            )
        else:
            print(
                "Sanity check:\n"
                "  - Does the enemy species id match the Pokemon on screen?\n"
                "    (Gen 1 internal ids are NOT the same as Pokedex numbers.\n"
                "     Look up the species in pret/pokered's constants/pokemon_constants.asm.)\n"
                "  - Does enemy HP look reasonable (positive, <= ~150 for early-route Pokemon)?\n"
                f"  - If in_battle == 2 (trainer), is (class={cls}, number={num}) consistent\n"
                "    with the trainer you're fighting?"
            )
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
