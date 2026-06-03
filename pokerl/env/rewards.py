"""Reward functions for PokemonRedEnv.

Reward design is versioned. Each version is a small, named class so we
can A/B them in experiments and document them in the writeup. v1 is
the first honest attempt: explore, level up, advance story flags, earn
badges. Expect v2+ to fix things v1 gets wrong.
"""
from __future__ import annotations

from typing import Protocol

from pokerl.env import ram_map as rm


class MemoryView(Protocol):
    def __getitem__(self, key): ...  # pragma: no cover


class Reward(Protocol):
    """A reward function tracks per-episode state and emits a scalar each step."""

    def reset(self, mem: MemoryView) -> None: ...
    def compute(self, mem: MemoryView) -> float: ...


class RewardV1:
    """First reward version: exploration + levels + flags + badges - step penalty.

    Tuned weights:
      +1   per newly-visited (map, x, y) tuple
      +5   per total levels gained across the party
      +10  per new event flag set
      +100 per badge earned
      -0.001 step penalty

    Baselines (level total, event flag count, badge bitfield) are captured
    on reset() so the first step doesn't pay the agent for state that
    already existed in the save state.
    """

    EXPLORE_REWARD = 1.0
    LEVEL_REWARD = 5.0
    FLAG_REWARD = 10.0
    BADGE_REWARD = 100.0
    STEP_PENALTY = -0.001

    def __init__(self) -> None:
        self._visited: set[tuple[int, int, int]] = set()
        self._base_level_total = 0
        self._base_flag_count = 0
        self._base_badge_count = 0

    def reset(self, mem: MemoryView) -> None:
        self._visited = set()
        # Capture baselines so deltas are zero at step 1.
        self._base_level_total = sum(rm.party_levels(mem))
        self._base_flag_count = rm.event_flags_popcount(mem)
        self._base_badge_count = rm.badges_count(mem)
        # Seed visited with the starting tile so the agent doesn't get
        # paid for "discovering" where it spawned.
        x, y = rm.player_position(mem)
        self._visited.add((rm.map_id(mem), x, y))

    def compute(self, mem: MemoryView) -> float:
        reward = self.STEP_PENALTY

        # Exploration
        x, y = rm.player_position(mem)
        key = (rm.map_id(mem), x, y)
        if key not in self._visited:
            self._visited.add(key)
            reward += self.EXPLORE_REWARD

        # Levels
        level_total = sum(rm.party_levels(mem))
        if level_total > self._base_level_total:
            reward += self.LEVEL_REWARD * (level_total - self._base_level_total)
            self._base_level_total = level_total

        # Event flags
        flag_count = rm.event_flags_popcount(mem)
        if flag_count > self._base_flag_count:
            reward += self.FLAG_REWARD * (flag_count - self._base_flag_count)
            self._base_flag_count = flag_count

        # Badges
        badge_count = rm.badges_count(mem)
        if badge_count > self._base_badge_count:
            reward += self.BADGE_REWARD * (badge_count - self._base_badge_count)
            self._base_badge_count = badge_count

        return reward

    # Debug introspection for logging / writeup
    @property
    def unique_tiles_visited(self) -> int:
        return len(self._visited)
