"""Pin RAM map reads against the canonical post-intro save state.

Skips if states/post_intro.state is missing — that file is derived from
the ROM and intentionally gitignored, so it only exists on machines where
make_save_state has been run.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROM = ROOT / "roms" / "pokemon_red.gb"
STATE = ROOT / "states" / "post_intro.state"


pytestmark = pytest.mark.skipif(
    not STATE.exists() or not ROM.exists(),
    reason="post_intro.state or ROM not present; run pokerl.scripts.make_save_state",
)


@pytest.fixture(scope="module")
def mem():
    from pyboy import PyBoy

    pyboy = PyBoy(str(ROM), window="null")
    with open(STATE, "rb") as f:
        pyboy.load_state(f)
    pyboy.tick()
    yield pyboy.memory
    pyboy.stop()


def test_party_count_is_one(mem):
    from pokerl.env import ram_map as rm
    assert rm.party_count(mem) == 1


def test_no_badges_yet(mem):
    from pokerl.env import ram_map as rm
    assert rm.badges_count(mem) == 0
    assert rm.badges_bitfield(mem) == 0


def test_starter_alive(mem):
    from pokerl.env import ram_map as rm
    levels = rm.party_levels(mem)
    hp = rm.party_hp(mem)
    assert len(levels) == 1
    assert 5 <= levels[0] <= 12, f"unexpected starter level: {levels[0]}"
    cur, mx = hp[0]
    assert cur > 0 and cur <= mx, f"weird HP read: {cur}/{mx}"


def test_in_pallet_or_nearby(mem):
    from pokerl.env import ram_map as rm
    assert rm.map_id(mem) in {
        rm.MAP_PALLET_TOWN,
        rm.MAP_OAKS_LAB,
        rm.MAP_ROUTE_1,
    }


def test_not_in_battle(mem):
    from pokerl.env import ram_map as rm
    assert rm.in_battle(mem) == 0
