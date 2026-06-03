"""Unit tests for RewardV1 using a dict-backed fake memory.

These tests don't need PyBoy or the ROM — they verify reward bookkeeping
in isolation. RAM-map address correctness is covered by test_ram_map.
"""
from __future__ import annotations

from pokerl.env import ram_map as rm
from pokerl.env.rewards import RewardV1


class FakeMem:
    """Minimal MemoryView shim backed by a dict."""

    def __init__(self, data: dict[int, int]) -> None:
        self.data = data

    def __getitem__(self, key):
        if isinstance(key, slice):
            return bytes(self.data.get(i, 0) for i in range(key.start, key.stop))
        return self.data.get(key, 0)


def base_state(
    *,
    party_count: int = 1,
    level: int = 6,
    badges: int = 0,
    map_id: int = 0,
    x: int = 8,
    y: int = 13,
) -> dict[int, int]:
    state: dict[int, int] = {}
    state[rm.ADDR_PARTY_COUNT] = party_count
    # one party mon with the given level
    state[rm.ADDR_PARTY_MON_BASE + rm.PMON_OFFSET_LEVEL] = level
    state[rm.ADDR_BADGES] = badges
    state[rm.ADDR_MAP_ID] = map_id
    state[rm.ADDR_PLAYER_X] = x
    state[rm.ADDR_PLAYER_Y] = y
    return state


def test_first_step_is_just_step_penalty():
    mem = FakeMem(base_state())
    r = RewardV1()
    r.reset(mem)
    # No state changed -> only the step penalty fires
    assert r.compute(mem) == r.STEP_PENALTY


def test_new_tile_pays_exploration_reward():
    state = base_state(x=8, y=13)
    mem = FakeMem(state)
    r = RewardV1()
    r.reset(mem)

    # Move to a new tile
    state[rm.ADDR_PLAYER_X] = 9
    got = r.compute(mem)
    assert got == r.STEP_PENALTY + r.EXPLORE_REWARD

    # Revisit -> no exploration reward
    got2 = r.compute(mem)
    assert got2 == r.STEP_PENALTY


def test_level_up_pays_per_level():
    state = base_state(level=6)
    mem = FakeMem(state)
    r = RewardV1()
    r.reset(mem)

    # Gain 2 levels in one step (e.g., big EXP gain)
    state[rm.ADDR_PARTY_MON_BASE + rm.PMON_OFFSET_LEVEL] = 8
    got = r.compute(mem)
    assert got == r.STEP_PENALTY + 2 * r.LEVEL_REWARD


def test_badge_pays_per_badge():
    state = base_state(badges=0)
    mem = FakeMem(state)
    r = RewardV1()
    r.reset(mem)

    # Earn Brock badge (bit 0)
    state[rm.ADDR_BADGES] = 0b0000_0001
    got = r.compute(mem)
    assert got == r.STEP_PENALTY + r.BADGE_REWARD


def test_baselines_prevent_paying_for_initial_state():
    # If the agent starts already at level 30 with 3 badges and a bunch of
    # flags, the first compute() should pay nothing for that pre-existing
    # state — only the step penalty.
    state = base_state(level=30, badges=0b0000_0111)
    state[rm.ADDR_EVENT_FLAGS_START] = 0xFF  # 8 flags already set
    mem = FakeMem(state)
    r = RewardV1()
    r.reset(mem)
    assert r.compute(mem) == r.STEP_PENALTY


def test_unique_tiles_visited_tracks_set_size():
    state = base_state(x=8, y=13)
    mem = FakeMem(state)
    r = RewardV1()
    r.reset(mem)
    assert r.unique_tiles_visited == 1  # seeded with start tile

    state[rm.ADDR_PLAYER_X] = 9
    r.compute(mem)
    state[rm.ADDR_PLAYER_Y] = 14
    r.compute(mem)
    assert r.unique_tiles_visited == 3
