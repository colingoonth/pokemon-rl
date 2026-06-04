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


class RewardV2:
    """V1 + movement bonus + battle-aware rewards.

    V1 failed because the agent learned to never press movement buttons.
    V2 fixes that AND adds combat signals (new encounters, beating
    Pokemon, winning/losing battles, fainting penalty). Per-episode
    sets so the agent rediscovers each episode rather than being
    permanently penalized for old runs.

    Reward components:
      Movement / exploration:
        EXPLORE_REWARD      +1     new (map_id, x, y) tuple
        MOVE_BONUS          +0.2   position changed but tile already visited
        STEP_PENALTY        -0.001 every step
      Progression:
        LEVEL_REWARD        +3/level    party-total levels gained
        FLAG_REWARD         +10/flag    new event flag set
        BADGE_REWARD        +100/badge  gym badge earned
      Battle:
        NEW_ENCOUNTER       +5     in_battle 0->nonzero with unseen identity
        BEAT_MON_REWARD     +2     enemy mon HP -> 0 during battle
        FAINT_PENALTY       -2     party mon HP -> 0 (per faint)
        WIN_BATTLE          +10    in_battle nonzero->0 with party alive
        LOSE_BATTLE         -7     in_battle nonzero->0 with whole party fainted

    The MOVE_BONUS does NOT apply during battle/dialogue because position
    is locked then — we don't want to penalize engaging NPCs, but we
    also don't want to reward "moved nowhere because the game is busy."
    """

    EXPLORE_REWARD = 1.0
    MOVE_BONUS = 0.2
    LEVEL_REWARD = 3.0
    FLAG_REWARD = 10.0
    BADGE_REWARD = 100.0
    STEP_PENALTY = -0.001
    NEW_ENCOUNTER = 5.0
    BEAT_MON_REWARD = 2.0
    FAINT_PENALTY = -2.0
    WIN_BATTLE = 10.0
    LOSE_BATTLE = -7.0

    def __init__(self) -> None:
        self._visited: set[tuple[int, int, int]] = set()
        self._last_position: tuple[int, int, int] | None = None
        self._base_level_total = 0
        self._base_flag_count = 0
        self._base_badge_count = 0
        # Battle state
        self._last_in_battle = 0
        self._last_enemy_hp = 0
        self._encountered_species: set[int] = set()
        self._encountered_trainers: set[tuple[int, int]] = set()
        # Per-party HP tracking for individual-faint detection
        self._last_party_hp: list[int] = []

    def reset(self, mem: MemoryView) -> None:
        self._visited = set()
        x, y = rm.player_position(mem)
        spawn = (rm.map_id(mem), x, y)
        self._visited.add(spawn)
        self._last_position = spawn
        self._base_level_total = sum(rm.party_levels(mem))
        self._base_flag_count = rm.event_flags_popcount(mem)
        self._base_badge_count = rm.badges_count(mem)
        self._last_in_battle = rm.in_battle(mem)
        self._last_enemy_hp = rm.enemy_mon_hp(mem) if self._last_in_battle else 0
        self._encountered_species = set()
        self._encountered_trainers = set()
        self._last_party_hp = [cur for cur, _mx in rm.party_hp(mem)]

    def compute(self, mem: MemoryView) -> float:
        reward = self.STEP_PENALTY

        # ----- Movement / exploration -----
        in_battle_now = rm.in_battle(mem)
        # Only credit movement when NOT in a battle (position is locked in battle).
        if in_battle_now == 0:
            x, y = rm.player_position(mem)
            position = (rm.map_id(mem), x, y)
            if self._last_position is not None and position != self._last_position:
                if position in self._visited:
                    reward += self.MOVE_BONUS
                else:
                    reward += self.EXPLORE_REWARD
                    self._visited.add(position)
            self._last_position = position

        # ----- Battle transitions -----
        battle_started = self._last_in_battle == 0 and in_battle_now != 0
        battle_ended = self._last_in_battle != 0 and in_battle_now == 0

        # Pre-read party HP once for both faint + battle-end logic
        party_hp = [cur for cur, _mx in rm.party_hp(mem)]

        if battle_started:
            if in_battle_now == 1:  # wild
                species = rm.enemy_mon_species(mem)
                if species not in self._encountered_species:
                    self._encountered_species.add(species)
                    reward += self.NEW_ENCOUNTER
            elif in_battle_now == 2:  # trainer
                tid = rm.trainer_id(mem)
                if tid not in self._encountered_trainers:
                    self._encountered_trainers.add(tid)
                    reward += self.NEW_ENCOUNTER

        # Beat-a-mon detection: enemy HP just dropped to 0 while still in battle
        if in_battle_now != 0 and self._last_in_battle != 0:
            enemy_hp_now = rm.enemy_mon_hp(mem)
            if self._last_enemy_hp > 0 and enemy_hp_now == 0:
                reward += self.BEAT_MON_REWARD
            self._last_enemy_hp = enemy_hp_now
        elif in_battle_now != 0 and battle_started:
            # Just entered a battle; capture initial enemy HP for next step delta
            self._last_enemy_hp = rm.enemy_mon_hp(mem)
        elif in_battle_now == 0:
            self._last_enemy_hp = 0

        if battle_ended:
            # Determine win vs lose based on party state at battle end
            party_alive = any(hp > 0 for hp in party_hp)
            if party_alive:
                reward += self.WIN_BATTLE
            else:
                reward += self.LOSE_BATTLE

        # ----- Per-faint penalty for party members -----
        # Compare element-wise against last party HP. Fire -2 per faint.
        # A "faint" is hp going from >0 to 0 on the same slot.
        for i in range(min(len(party_hp), len(self._last_party_hp))):
            if self._last_party_hp[i] > 0 and party_hp[i] == 0:
                reward += self.FAINT_PENALTY
        self._last_party_hp = party_hp

        # ----- Progression: levels, flags, badges (unchanged from V1 logic) -----
        level_total = sum(rm.party_levels(mem))
        if level_total > self._base_level_total:
            reward += self.LEVEL_REWARD * (level_total - self._base_level_total)
            self._base_level_total = level_total

        flag_count = rm.event_flags_popcount(mem)
        if flag_count > self._base_flag_count:
            reward += self.FLAG_REWARD * (flag_count - self._base_flag_count)
            self._base_flag_count = flag_count

        badge_count = rm.badges_count(mem)
        if badge_count > self._base_badge_count:
            reward += self.BADGE_REWARD * (badge_count - self._base_badge_count)
            self._base_badge_count = badge_count

        self._last_in_battle = in_battle_now
        return reward

    @property
    def unique_tiles_visited(self) -> int:
        return len(self._visited)


class RewardV2_1(RewardV2):
    """V2 with exploration rewards scaled down to relatively-boost combat.

    V2's first ELSA run showed +700 returns dominated by exploration
    rewards (32 envs visited 5500+ tiles in 80k steps). Combat signals
    (+5 encounter, +10 win, -7 lose, +100 badge) were correctly firing
    but vastly outweighed by exploration. V2.1 cuts exploration ~3x so
    combat has a comparable pull on the policy.
    """

    EXPLORE_REWARD = 0.3   # was 1.0
    MOVE_BONUS = 0.05      # was 0.2


REWARD_REGISTRY: dict[str, type] = {
    "RewardV1": RewardV1,
    "RewardV2": RewardV2,
    "RewardV2_1": RewardV2_1,
}


def get_reward_cls(name: str) -> type:
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward class {name!r}. "
            f"Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]
