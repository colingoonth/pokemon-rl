"""Play Pokemon Red interactively; save state on quit.

Default writes to states/post_intro.state — the canonical training
start. Pass --out to save somewhere else (useful for capturing
specific game states like an in-battle save for RAM verification).
"""
import argparse
from pathlib import Path

from pyboy import PyBoy

ROOT = Path(__file__).resolve().parents[2]
ROM = ROOT / "roms" / "pokemon_red.gb"
DEFAULT_STATE = ROOT / "states" / "post_intro.state"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_STATE,
                   help="Output path for the save state (default: states/post_intro.state).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    STATE = args.out
    if not ROM.exists():
        raise SystemExit(f"ROM not found at {ROM}")

    print("Opening Pokemon Red in an SDL2 window.")
    print()
    print("Controls (PyBoy default keymap):")
    print("  Arrow keys = D-pad")
    print("  A          = A button")
    print("  S          = B button")
    print("  Enter      = Start")
    print("  Backspace  = Select")
    print()
    print("Play through the Oak intro:")
    print("  - Name yourself + your rival")
    print("  - Pick a starter")
    print("  - Step outside until you're standing in Pallet Town with the starter")
    print()
    print(f"When you're at a good starting state, close the window.")
    print(f"State will be saved to: {STATE}")
    print()

    pyboy = PyBoy(str(ROM), window="SDL2")
    try:
        while pyboy.tick():
            pass
        STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE, "wb") as f:
            pyboy.save_state(f)
        print(f"\nSaved state -> {STATE}")
        print(f"Size: {STATE.stat().st_size:,} bytes")
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
