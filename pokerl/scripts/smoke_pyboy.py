"""Smoke test: load the ROM, advance frames, dump a screenshot."""
from pathlib import Path

import imageio.v3 as iio
from pyboy import PyBoy

ROOT = Path(__file__).resolve().parents[2]
ROM = ROOT / "roms" / "pokemon_red.gb"
OUT = ROOT / "notes" / "smoke.png"
FRAMES = 600


def main() -> None:
    if not ROM.exists():
        raise SystemExit(f"ROM not found at {ROM}")

    pyboy = PyBoy(str(ROM), window="null")
    try:
        for _ in range(FRAMES):
            pyboy.tick()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(OUT, pyboy.screen.ndarray)
    finally:
        pyboy.stop()

    print(f"smoke OK after {FRAMES} frames -> {OUT}")


if __name__ == "__main__":
    main()
