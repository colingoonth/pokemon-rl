"""Load states/post_intro.state and print every RAM map read.

The point: sanity-check our addresses against ground truth from a real
playthrough. If party_count != 1 or badges != 0, an address is wrong
and we want to know NOW, not 100K training steps in.
"""
from pathlib import Path

from pyboy import PyBoy

from pokerl.env import ram_map as rm

ROOT = Path(__file__).resolve().parents[2]
ROM = ROOT / "roms" / "pokemon_red.gb"
STATE = ROOT / "states" / "post_intro.state"


SPECIES_NAMES = {
    0x01: "Bulbasaur",
    0x04: "Charmander",
    0x07: "Squirtle",
    0x99: "Bulbasaur (dex#1, internal varies)",  # placeholder; internal ids differ from dex
}


def main() -> None:
    if not STATE.exists():
        raise SystemExit(
            f"No save state at {STATE}. Run: uv run python -m pokerl.scripts.make_save_state"
        )

    pyboy = PyBoy(str(ROM), window="null")
    try:
        with open(STATE, "rb") as f:
            pyboy.load_state(f)
        # advance one tick to ensure memory reflects loaded state
        pyboy.tick()

        mem = pyboy.memory

        print("=== Post-intro save state RAM readings ===")
        print()
        print(f"Party count:        {rm.party_count(mem)}")
        print(f"Party species (raw): {[hex(s) for s in rm.party_species(mem)]}")
        print(f"Party levels:       {rm.party_levels(mem)}")
        print(f"Party HP:           {rm.party_hp(mem)}")
        print()
        print(f"Badges bitfield:    {rm.badges_bitfield(mem):#010b}")
        print(f"Badge count:        {rm.badges_count(mem)}")
        print()
        print(f"Map ID:             {rm.map_id(mem)} (0x{rm.map_id(mem):02X})")
        print(f"Player position:    {rm.player_position(mem)}")
        print()
        print(f"Money:              ${rm.money(mem)}")
        print(f"In battle:          {rm.in_battle(mem)}")
        print(f"Event flags set:    {rm.event_flags_popcount(mem)}")
        print()
        print("=== Sanity checks (should all PASS) ===")

        checks = [
            ("party_count == 1", rm.party_count(mem) == 1),
            ("badges == 0", rm.badges_count(mem) == 0),
            ("party_levels[0] in 5..10", 5 <= rm.party_levels(mem)[0] <= 10),
            ("not in battle", rm.in_battle(mem) == 0),
            ("map_id likely Pallet/Lab/Route1",
             rm.map_id(mem) in {rm.MAP_PALLET_TOWN, rm.MAP_OAKS_LAB, rm.MAP_ROUTE_1, 0x25}),
        ]
        all_pass = True
        for name, ok in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}")
            if not ok:
                all_pass = False
        print()
        print("OVERALL:", "all reads look correct" if all_pass else "AT LEAST ONE ADDRESS IS WRONG")
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
