"""Reward functions for PokemonRedEnv.

Reward design is versioned. Each version is a small, named class so we
can A/B them in experiments and document them in the writeup. v1 is
the first honest attempt: explore, level up, advance story flags, earn
badges. Expect v2+ to fix things v1 gets wrong.
"""
from __future__ import annotations

import math
from typing import Protocol

from pokerl.env import ram_map as rm


class MemoryView(Protocol):
    def __getitem__(self, key): ...  # pragma: no cover


class Reward(Protocol):
    """A reward function tracks per-episode state and emits a scalar each step."""

    def reset(self, mem: MemoryView) -> None: ...
    def compute(self, mem: MemoryView) -> float: ...


class RewardV0_1:
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


class RewardV0_2:
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


class RewardV0_2_1(RewardV0_2):
    """V2 with exploration rewards scaled down to relatively-boost combat.

    V2's first ELSA run showed +700 returns dominated by exploration
    rewards (32 envs visited 5500+ tiles in 80k steps). Combat signals
    (+5 encounter, +10 win, -7 lose, +100 badge) were correctly firing
    but vastly outweighed by exploration. V2.1 cuts exploration ~3x so
    combat has a comparable pull on the policy.
    """

    EXPLORE_REWARD = 0.3   # was 1.0
    MOVE_BONUS = 0.05      # was 0.2


class RewardV0_2_2(RewardV0_2_1):
    """V0.2.1 + fix flee-as-win exploit + new-map / catch rewards.

    V0.2.1 had a bug: WIN_BATTLE fired any time in_battle went from
    nonzero -> 0 with the party alive, which included fleeing. The
    agent learned to walk into grass, encounter, flee, repeat — earning
    +10 per flee. Visible in watch.py: agent entered Rattata battle,
    fled immediately, repeated.

    V0.2.2 fixes that by tracking enemy_killed_this_battle and
    pokemon_caught_this_battle during the battle, then only awarding
    WIN_BATTLE when one of those flags is set. Fleeing now pays
    FLEE_PENALTY (-1) — small enough that strategic fleeing (low HP,
    overmatched) is still cheaper than -7 lose + faint penalties, but
    expensive enough that grind-flee is unprofitable.

    Two new positive signals to pull the agent out of "grind battles
    forever" once it converges on combat:
      NEW_MAP_REWARD (+5)        when entering a previously-unseen map_id
      CATCH_REWARD (+5)          per Pokemon caught (party_count++)
      NEW_CATCH_BONUS (+20)      additional bonus when the caught species
                                 is new to the per-episode caught set
    """

    BEAT_MON_REWARD = 10.0       # was +2 in V0.2 — bumped: this IS the wild-win reward
    TRAINER_WIN_BONUS = 20.0     # bonus for clearing a trainer battle (on top of per-kill)
    FLEE_PENALTY = -1.0
    NEW_MAP_REWARD = 5.0
    CATCH_REWARD = 5.0
    NEW_CATCH_BONUS = 20.0

    def __init__(self) -> None:
        super().__init__()
        self._enemy_killed_this_battle = False
        self._pokemon_caught_this_battle = False
        self._battle_was_trainer = False
        self._last_party_count = 0
        self._visited_maps: set[int] = set()
        self._caught_species: set[int] = set()

    def reset(self, mem: MemoryView) -> None:
        super().reset(mem)
        self._enemy_killed_this_battle = False
        self._pokemon_caught_this_battle = False
        self._battle_was_trainer = False
        self._last_party_count = rm.party_count(mem)
        self._visited_maps = {rm.map_id(mem)}
        self._caught_species = set()

    def _attr_v022(self, name: str, value: float) -> None:
        """Defensive: only attributes if a V0.4.0+ instance set up
        last_components. Calling on a plain V0_2_2 instance is a no-op.
        """
        components = getattr(self, "last_components", None)
        if components is not None and value != 0.0:
            components[name] = components.get(name, 0.0) + value

    def compute(self, mem: MemoryView) -> float:
        reward = self.STEP_PENALTY

        # ----- Movement / exploration -----
        in_battle_now = rm.in_battle(mem)
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

        # ----- New map_id -----
        current_map = rm.map_id(mem)
        if current_map not in self._visited_maps:
            self._visited_maps.add(current_map)
            reward += self.NEW_MAP_REWARD

        # ----- Battle transitions -----
        battle_started = self._last_in_battle == 0 and in_battle_now != 0
        battle_ended = self._last_in_battle != 0 and in_battle_now == 0

        party_hp = [cur for cur, _mx in rm.party_hp(mem)]
        party_count_now = rm.party_count(mem)

        if battle_started:
            self._enemy_killed_this_battle = False
            self._pokemon_caught_this_battle = False
            self._battle_was_trainer = (in_battle_now == 2)
            if in_battle_now == 1:
                species = rm.enemy_mon_species(mem)
                if species not in self._encountered_species:
                    self._encountered_species.add(species)
                    reward += self.NEW_ENCOUNTER
            elif in_battle_now == 2:
                tid = rm.trainer_id(mem)
                if tid not in self._encountered_trainers:
                    self._encountered_trainers.add(tid)
                    reward += self.NEW_ENCOUNTER

        if in_battle_now != 0 and self._last_in_battle != 0:
            enemy_hp_now = rm.enemy_mon_hp(mem)
            if self._last_enemy_hp > 0 and enemy_hp_now == 0:
                reward += self.BEAT_MON_REWARD
                self._enemy_killed_this_battle = True
            self._last_enemy_hp = enemy_hp_now
            if party_count_now > self._last_party_count:
                self._pokemon_caught_this_battle = True
        elif in_battle_now != 0 and battle_started:
            self._last_enemy_hp = rm.enemy_mon_hp(mem)
            if party_count_now > self._last_party_count:
                self._pokemon_caught_this_battle = True
        elif in_battle_now == 0:
            self._last_enemy_hp = 0

        # ----- Catch rewards (any time party_count increases) -----
        if party_count_now > self._last_party_count:
            n_new = party_count_now - self._last_party_count
            reward += self.CATCH_REWARD * n_new
            party_species = rm.party_species(mem)
            for slot in range(self._last_party_count, party_count_now):
                if slot < len(party_species):
                    species_id = party_species[slot]
                    if species_id not in self._caught_species:
                        self._caught_species.add(species_id)
                        reward += self.NEW_CATCH_BONUS
        self._last_party_count = party_count_now

        # ----- Battle-end resolution -----
        # Wild kills are paid per-mon via BEAT_MON_REWARD; there's no extra
        # "win" bonus for wild. Trainer wins get TRAINER_WIN_BONUS on top of
        # the per-mon kills. Catching pays its own CATCH_REWARD/NEW_CATCH_BONUS
        # earlier and does NOT count as a "win" so the agent isn't pushed to
        # catch every Pokemon in sight.
        if battle_ended:
            party_alive = any(hp > 0 for hp in party_hp)
            if not party_alive:
                reward += self.LOSE_BATTLE
                self._attr_v022("LOSE_BATTLE", self.LOSE_BATTLE)
            elif self._enemy_killed_this_battle and self._battle_was_trainer:
                reward += self.TRAINER_WIN_BONUS
                self._attr_v022("TRAINER_WIN_BONUS", self.TRAINER_WIN_BONUS)
            elif self._enemy_killed_this_battle or self._pokemon_caught_this_battle:
                pass  # already paid via BEAT_MON_REWARD or CATCH_REWARD
            else:
                reward += self.FLEE_PENALTY
                self._attr_v022("FLEE_PENALTY", self.FLEE_PENALTY)

        # ----- Per-faint penalty -----
        for i in range(min(len(party_hp), len(self._last_party_hp))):
            if self._last_party_hp[i] > 0 and party_hp[i] == 0:
                reward += self.FAINT_PENALTY
                self._attr_v022("FAINT_PENALTY", self.FAINT_PENALTY)
        self._last_party_hp = party_hp

        # ----- Progression -----
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


class RewardV0_2_3(RewardV0_2_2):
    """V0.2.2 + Pokemon Center heal rewards + harsh faint penalty.

    V0.2.2 had a blackout-as-free-heal exploit: agent learns the loop
    "explore, take damage, faint, respawn fully healed at last PC, repeat."
    Faint penalty (-2) was dwarfed by per-life exploration gain (+200-300).

    V0.2.3 makes blackout strictly worse than the PC route:
      PC_HEAL_LOW    +2    healed at PC when party_hp < 80% OR any mon < 50%
      PC_HEAL_FULL   -1    healed at PC at near-full HP (anti-spam)
      PC_FIRST_VISIT +5    first time stepping into a given PC map_id
                           (stacks with PC_HEAL_* on first-time real heal)
      FAINT_PENALTY  -250  (was -2). Must dominate per-life exploration gain.

    Blackout guard: when the game whites out and force-heals the party at
    the last-visited PC (or home), the heal-event detection would otherwise
    fire PC_HEAL_LOW + PC_FIRST_VISIT and partially refund the -250. We
    set _blackout_pending=True when LOSE_BATTLE triggers, suppress the
    next heal/visit reward, then clear the flag.
    """

    FAINT_PENALTY = -250.0
    PC_HEAL_LOW = 2.0
    PC_HEAL_FULL = -1.0
    PC_FIRST_VISIT = 5.0
    LOW_HP_PARTY_THRESHOLD = 0.80
    LOW_HP_ANY_THRESHOLD = 0.50

    def __init__(self) -> None:
        super().__init__()
        self._visited_pokecenters: set[int] = set()
        self._blackout_pending = False
        self._last_party_hp_max: list[int] = []

    def reset(self, mem: MemoryView) -> None:
        super().reset(mem)
        self._visited_pokecenters = set()
        if rm.map_id(mem) in rm.POKECENTER_MAP_IDS:
            self._visited_pokecenters.add(rm.map_id(mem))
        self._blackout_pending = False
        self._last_party_hp_max = [mx for _cur, mx in rm.party_hp(mem)]

    def compute(self, mem: MemoryView) -> float:
        # Snapshot prev state BEFORE super() overwrites self._last_party_hp
        prev_cur = list(self._last_party_hp)
        prev_max = list(self._last_party_hp_max)

        party_hp = rm.party_hp(mem)
        cur_hp = [c for c, _m in party_hp]
        max_hp = [m for _c, m in party_hp]

        current_map = rm.map_id(mem)
        on_pc_map = current_map in rm.POKECENTER_MAP_IDS

        prev_total = sum(prev_cur)
        prev_total_max = sum(prev_max) if prev_max else 0
        was_below_max = prev_max and prev_total < prev_total_max
        all_at_max = len(party_hp) > 0 and all(
            c == m for c, m in party_hp if m > 0
        )
        heal_event_now = bool(on_pc_map and was_below_max and all_at_max)
        first_visit_now = on_pc_map and current_map not in self._visited_pokecenters

        # Delegate to parent for base reward (movement, battles, faints, etc.)
        # FAINT_PENALTY override propagates via self lookup.
        reward = super().compute(mem)

        if heal_event_now or first_visit_now:
            if self._blackout_pending:
                # Suppress refund from blackout-triggered heal/visit
                if first_visit_now:
                    self._visited_pokecenters.add(current_map)
                self._blackout_pending = False
            else:
                if first_visit_now:
                    self._visited_pokecenters.add(current_map)
                    reward += self.PC_FIRST_VISIT
                if heal_event_now:
                    if prev_total_max > 0:
                        prev_pct = prev_total / prev_total_max
                    else:
                        prev_pct = 1.0
                    any_low = any(
                        (pc / pm) < self.LOW_HP_ANY_THRESHOLD
                        for pc, pm in zip(prev_cur, prev_max)
                        if pm > 0
                    )
                    if prev_pct < self.LOW_HP_PARTY_THRESHOLD or any_low:
                        reward += self.PC_HEAL_LOW
                    else:
                        reward += self.PC_HEAL_FULL

        # Detect blackout trigger: battle just ended with the whole party dead.
        # Parent already updated _last_in_battle, so use the snapshotted prev_cur.
        was_alive_last = any(c > 0 for c in prev_cur) if prev_cur else False
        all_dead_now = len(cur_hp) > 0 and all(c == 0 for c in cur_hp)
        in_battle_now = rm.in_battle(mem)
        if in_battle_now == 0 and all_dead_now and was_alive_last:
            self._blackout_pending = True

        self._last_party_hp_max = max_hp
        return reward


class RewardV0_2_4_f25(RewardV0_2_3):
    """V0.2.3 with FAINT_PENALTY relaxed to -25.

    Watching V0.2.3 (14376, 14430) showed the agent refusing to engage in
    battles at all — -250 made the expected value of any uncertain fight
    catastrophically negative. Death spiral: no battles -> no learning ->
    no confidence -> still no battles.

    V0.2.4 family sweeps the faint magnitude to find the smallest value
    that still dominates per-life exploration gain (~+200) without
    triggering the death spiral. At -25, an 80%-confidence battle has
    EV = 0.8*10 + 0.2*(-7 + -25) = +1.6 -- positive, so the agent has
    a gradient signal toward fighting. At -50 and -100 the breakeven
    confidence threshold rises but stays well below V0.2.3's ~96%.
    """

    FAINT_PENALTY = -25.0


class RewardV0_2_4_f50(RewardV0_2_3):
    """V0.2.3 with FAINT_PENALTY = -50."""

    FAINT_PENALTY = -50.0


class RewardV0_2_4_f100(RewardV0_2_3):
    """V0.2.3 with FAINT_PENALTY = -100."""

    FAINT_PENALTY = -100.0


class RewardV0_2_5(RewardV0_2_4_f25):
    """V0.2.4_f25 + aggressive building / Pokemart / PokeCenter bonuses.

    Watching V0.2.4_f25 (the one variant whose agent actually engages in
    battles) showed the policy refuses to enter buildings or interact
    with NPCs in the overworld. It learned A-press = wasted ticks. For
    V1 (Brock), the agent MUST enter Brock's Gym (a building map_id).

    V0.2.5 makes building entry strongly rewarding:
      NEW_MAP_REWARD     overridden by subclasses (sweep magnitudes)
      MART_PC_BONUS      +20 extra when the new map is a Pokemart or PC
      PC_HEAL_LOW        bumped to +5 (was +2) — heal incentive matches
                         the bigger map exploration scale

    Dialog detection (rewarding NPC interactions specifically) is
    deferred to V0.2.6 pending verification of the right text-box-active
    RAM byte — that wasn't ready in time for tonight's sweep.

    Inherits everything else from V0.2.4_f25: FAINT_PENALTY -25, PC heal
    logic, blackout guard, flee penalty, battle rewards.
    """

    MART_PC_BONUS = 20.0
    PC_HEAL_LOW = 5.0  # was 2.0 in V0.2.3

    def compute(self, mem: MemoryView) -> float:
        # Snapshot BEFORE super updates _visited_maps so we can detect
        # the "this step entered a Mart/PC for the first time" event.
        current_map = rm.map_id(mem)
        is_new_map = current_map not in self._visited_maps
        is_mart_or_pc = (
            current_map in rm.POKECENTER_MAP_IDS
            or current_map in rm.POKEMART_MAP_IDS
        )

        reward = super().compute(mem)

        if is_new_map and is_mart_or_pc:
            reward += self.MART_PC_BONUS

        return reward


class RewardV0_2_5_b20(RewardV0_2_5):
    """V0.2.5 with NEW_MAP_REWARD = +20. First Mart/PC visit = +40 stacked."""

    NEW_MAP_REWARD = 20.0


class RewardV0_2_5_b50(RewardV0_2_5):
    """V0.2.5 with NEW_MAP_REWARD = +50. First Mart/PC visit = +70 stacked."""

    NEW_MAP_REWARD = 50.0


class RewardV0_2_5_b100(RewardV0_2_5):
    """V0.2.5 with NEW_MAP_REWARD = +100. First Mart/PC visit = +120 stacked."""

    NEW_MAP_REWARD = 100.0


class RewardV0_2_6(RewardV0_2_4_f25):
    """V0.2.4_f25 + small Mart/PC + Gym bonuses, restoring f25's balance.

    Watching the V0.2.5 sweep (b20/b50/b100) revealed a regression: the
    aggressive +20-to-+100 NEW_MAP_REWARD bonuses caused a new local
    optimum — agents converged on "flee every battle, explore new maps
    for the jackpots." V0.2.4_f25 (NEW_MAP +5) had organic strategic
    combat (fled Pidgey for type disadvantage, fought others); even
    V0.2.5_b20's +20 was enough to flip the EV math toward flee.

    V0.2.6 keeps f25's exact reward balance and only ADDS three small
    signals:
      MART_PC_BONUS  +10  Mart or PokeCenter first-visit
      GYM_BONUS      +20  Gym first-visit (Pewter Gym is V1 target)
      entropy_coef   0.03 (PPO config, not class const) -- between
                          f25's collapse-prone 0.01 and V0.2.5's
                          commit-prevention 0.1

    Both bonuses are small enough (relative to +10 BEAT_MON) that they
    shouldn't flip the optimal strategy away from fighting, but big
    enough to give a clear gradient toward critical locations.

    Brock's Gym (PEWTER_GYM = 0x36) gets +20 + +5 NEW_MAP = +25 on
    first visit. That's still less than killing a single trainer
    (+10 per kill + +20 TRAINER_WIN bonus = +50 for a 3-mon trainer
    battle), so combat stays the dominant strategy on-route while
    being weighted by the destination.

    Dialog detection reward (rewarding A-press on NPCs broadly) is
    deferred to V0.2.7 pending RAM-byte verification.
    """

    MART_PC_BONUS = 10.0
    GYM_BONUS = 20.0

    def compute(self, mem: MemoryView) -> float:
        current_map = rm.map_id(mem)
        is_new_map = current_map not in self._visited_maps
        is_mart_or_pc = (
            current_map in rm.POKECENTER_MAP_IDS
            or current_map in rm.POKEMART_MAP_IDS
        )
        is_gym = current_map in rm.GYM_MAP_IDS

        reward = super().compute(mem)

        if is_new_map:
            if is_mart_or_pc:
                reward += self.MART_PC_BONUS
            if is_gym:
                reward += self.GYM_BONUS

        return reward


class RewardV0_2_7(RewardV0_2_6):
    """V0.2.6 + bumped combat rewards to lower the fight-EV threshold.

    V0.2.6 at iter 2300 (~19M steps) still showed flee-everything
    behavior. Math: with BEAT_MON +10 and FAINT -25, EV(fight) > EV(flee)
    requires P(win) > 74% — a high bar for a partially-trained policy.
    Bumping BEAT_MON to +20 drops the breakeven win confidence to 60%,
    which is reachable for a moderate policy after a handful of wild
    battle samples. TRAINER_WIN_BONUS +20 -> +30 makes trainer battles
    (the V1 critical path on Route 2) more attractive than wild ones.

    Inherits everything else from V0.2.6: f25 balance, +10 Mart/PC
    bonus, +20 Pewter Gym bonus.

    NOTE: V0.2.7 was originally penciled for dialog detection. That's
    now V0.2.8 (still pending RAM-byte verification).
    """

    BEAT_MON_REWARD = 20.0      # was 10 in V0.2.2-V0.2.6
    TRAINER_WIN_BONUS = 30.0    # was 20 in V0.2.2-V0.2.6


class RewardV0_3_1(RewardV0_2_7):
    """V0.2.7 + bumped FLEE_PENALTY for the curriculum fork.

    V0.3.0 (curriculum start at Blue rival fight) showed the agent
    correctly EV-fleeing fights below the 60% breakeven — rational
    given the reward landscape, but stalls combat learning on the
    wild encounters Squirtle would lose. Bumping FLEE_PENALTY from
    -1 to -5 drops the fight-vs-flee breakeven from 60% to ~45%,
    pushing the agent to engage borderline matchups without changing
    in-battle action selection (Tackle vs Tail Whip vs status moves
    are unaffected — only the leave-battle action is penalized).

    Inherits everything else from V0.2.7.
    """

    FLEE_PENALTY = -5.0     # was -1 in V0.2.2-V0.2.7


class RewardV0_3_3(RewardV0_3_1):
    """V0.3.1 + quadratic HP-restored heal reward; disables threshold PC_HEAL_*.

    Replaces the discrete PC_HEAL_LOW / PC_HEAL_FULL bonuses (which only
    fire on transition-to-full at a PokeCenter) with a continuous signal:
    every step where party HP fraction increases, reward
        (delta_hp_frac)^2 * HEAL_QUAD_COEF
    Auto-scales: full heal from 10% HP pays ~0.81 * 12 = +9.7, full heal
    from 90% HP pays ~0.01 * 12 = +0.12. No magnitude knob to guess; the
    severity of the heal sets the size of the reward.

    Inspired by Whidden's PokemonRedExperiments v2 reward shape (uses
    coef 10); we use 12 here for slightly stronger pull. The quadratic
    fires anywhere HP rises — potion use mid-battle, nurse heal at PC,
    Pokemon Centers in subsequent cities — without us having to hand-
    define "heal events."

    Blackout force-heal is still suppressed via the existing
    _blackout_pending guard (snapshot before super() clears it).

    Inherits everything else from V0.3.1.
    """

    PC_HEAL_LOW = 0.0       # disabled — superseded by quadratic term
    PC_HEAL_FULL = 0.0      # disabled
    HEAL_QUAD_COEF = 12.0

    def __init__(self) -> None:
        super().__init__()
        self._last_hp_frac = 0.0

    def reset(self, mem: MemoryView) -> None:
        super().reset(mem)
        party_hp = rm.party_hp(mem)
        cur = sum(c for c, _m in party_hp)
        mx = sum(m for _c, m in party_hp)
        self._last_hp_frac = (cur / mx) if mx > 0 else 0.0

    def compute(self, mem: MemoryView) -> float:
        # Snapshot blackout flag + in_battle BEFORE super().compute()
        # clears them. The flag-based guard catches multi-step blackout
        # sequences. The single-step guard below catches the case where
        # the agent faints AND respawns within one env step — by the
        # time super() runs, party HP is already full, so the V0_2_3
        # `all_dead_now` snapshot never triggers _blackout_pending.
        was_blackout_pending = self._blackout_pending
        prev_in_battle = self._last_in_battle

        reward = super().compute(mem)

        party_hp = rm.party_hp(mem)
        cur = sum(c for c, _m in party_hp)
        mx = sum(m for _c, m in party_hp)
        hp_frac = (cur / mx) if mx > 0 else 0.0
        delta = max(0.0, hp_frac - self._last_hp_frac)

        # Single-step blackout signature: battle just ended this step,
        # agent is now on a PC map at full HP. This is a healed respawn —
        # not a player-initiated PC heal (which happens with no battle
        # transition) and not in-battle healing (which doesn't end battle).
        in_battle_now = rm.in_battle(mem)
        on_pc_map = rm.map_id(mem) in rm.POKECENTER_MAP_IDS
        single_step_blackout = (
            prev_in_battle != 0 and in_battle_now == 0
            and on_pc_map and hp_frac >= 1.0
        )

        if (delta > 0 and self.HEAL_QUAD_COEF > 0
                and not was_blackout_pending and not single_step_blackout):
            heal_val = (delta ** 2) * self.HEAL_QUAD_COEF
            reward += heal_val
            self._attr_v022("HEAL_QUAD", heal_val)
        self._last_hp_frac = hp_frac
        return reward


class RewardV0_3_4(RewardV0_3_3):
    """V0.3.4: full strip + curve-based future-seeking rewards.

    Philosophy shift: remove direct, dense action rewards; replace with
    log / power-law / capped rewards on game state. Force the agent to
    seek future value (badges, exploration, pokedex collection) rather
    than maximizing per-step combat.

    Active rewards:
      - HEAL_QUAD_COEF 15: (delta_hp_frac)^2 * 15 (was 12 in V0.3.3)
      - LEVEL: log_6(level_just_hit) per unit increase
      - CATCH: log_2.5(new_pokedex_count) per pokedex increment
      - EXPLORE: 0.0001 * N^1.001 per new tile (N = total seen so far)
      - BEAT_MON: capped 5/area, falling log_2.5(7 - kill_n)
      - BADGE: 15 * x^1.3 per badge gained (x = new total)
      - POKEBALL_BUY: 5 * max(0, 1 - n/20) per ball bought
      - STUCK: -0.005/step after 200 steps with no new tile discovered
      - TRAINER_WIN_BONUS +10 (down from +30)
      - FLEE_PENALTY -0.5 (down from -5)
      - FAINT_PENALTY -5 (down from -25 — Whidden-style softening)
      - LOSE_BATTLE -20 (up from -7)

    Disabled (set to 0): NEW_MAP_REWARD, MOVE_BONUS, STEP_PENALTY,
      MART_PC_BONUS, PC_FIRST_VISIT, GYM_BONUS, FLAG_REWARD,
      NEW_CATCH_BONUS, NEW_ENCOUNTER, BEAT_MON_REWARD (replaced by capped log),
      LEVEL_REWARD (replaced by log_6), CATCH_REWARD (replaced by log_2.5),
      EXPLORE_REWARD (replaced by curve), BADGE_REWARD (replaced by power law).
    """

    # --- Disable inherited direct rewards (replaced by curves below) ---
    EXPLORE_REWARD = 0.0
    MOVE_BONUS = 0.0
    STEP_PENALTY = 0.0
    LEVEL_REWARD = 0.0
    FLAG_REWARD = 0.0
    BADGE_REWARD = 0.0
    NEW_ENCOUNTER = 0.0
    BEAT_MON_REWARD = 0.0
    NEW_MAP_REWARD = 0.0
    CATCH_REWARD = 0.0
    NEW_CATCH_BONUS = 0.0
    MART_PC_BONUS = 0.0
    PC_FIRST_VISIT = 0.0
    GYM_BONUS = 0.0
    PC_HEAL_LOW = 0.0
    PC_HEAL_FULL = 0.0

    # --- Direct constants kept but retuned ---
    FAINT_PENALTY = -5.0
    LOSE_BATTLE = -20.0
    FLEE_PENALTY = -0.5
    TRAINER_WIN_BONUS = 10.0
    HEAL_QUAD_COEF = 15.0

    # --- New curve constants ---
    KILL_CAP_PER_AREA = 5
    EXPLORE_COEF = 0.0001
    EXPLORE_EXPONENT = 1.001
    POKEBALL_BUY_COEF = 5.0
    POKEBALL_BUY_CAP = 20
    STUCK_NO_TILE_THRESHOLD = 200
    STUCK_PENALTY_PER_STEP = -0.005
    BADGE_COEF = 15.0
    BADGE_EXPONENT = 1.3

    def __init__(self) -> None:
        super().__init__()
        self._kills_per_area: dict[int, int] = {}
        self._last_pokedex_count = 0
        self._last_pokeballs = 0
        self._steps_since_new_tile = 0
        # Reward attribution: per-step contribution breakdown.
        # Watchers (eval/watch.py) read this after each compute() to
        # aggregate which reward components fired. Reset at the top of
        # every compute().
        self.last_components: dict[str, float] = {}

    def reset(self, mem) -> None:
        super().reset(mem)
        self._kills_per_area = {}
        self._last_pokedex_count = rm.pokedex_owned_count(mem)
        self._last_pokeballs = rm.bag_item_quantity(mem, rm.ITEM_POKEBALL_ID)
        self._steps_since_new_tile = 0
        self.last_components = {}

    def _track(self, name: str, value: float) -> float:
        """Record a reward contribution under `name` and return it
        unchanged. Use as: `reward += self._track("EXPLORE", coef * x)`.
        """
        if value != 0.0:
            self.last_components[name] = (
                self.last_components.get(name, 0.0) + value
            )
        return value

    def compute(self, mem) -> float:
        # Reset attribution dict for this step. Parent compute() runs
        # before our local tracking, so anything pre-V0_3_4 goes into
        # INHERITED (a single bucket). Tracked components below add to
        # last_components individually.
        self.last_components = {}

        # Snapshot pre-state for delta-based custom rewards. super() will
        # update these in-place during its compute pass, so we need values
        # from BEFORE the call.
        prev_level_total = self._base_level_total
        prev_badge_count = self._base_badge_count
        prev_visited_count = len(self._visited)
        prev_enemy_hp = self._last_enemy_hp
        prev_in_battle = self._last_in_battle

        reward = super().compute(mem)
        # super() has already written its tracked components (HEAL_QUAD,
        # LOSE_BATTLE, FAINT_PENALTY, TRAINER_WIN_BONUS, FLEE_PENALTY) via
        # _attr_v022. Anything in super's scalar that wasn't tracked goes
        # into UNATTRIBUTED (covers older V0.1/V0.2 rewards that aren't
        # instrumented because they're zeroed in the V0.4 line).
        tracked_so_far = sum(self.last_components.values())
        unattrib = reward - tracked_so_far
        if abs(unattrib) > 1e-9:
            self.last_components["UNATTRIBUTED"] = unattrib

        # ---- LEVEL: log_6(new_level) per unit gained ----
        level_total_now = sum(rm.party_levels(mem))
        if level_total_now > prev_level_total:
            for new_level in range(prev_level_total + 1, level_total_now + 1):
                if new_level > 1:
                    reward += self._track(
                        "LEVEL", math.log(new_level) / math.log(6)
                    )

        # ---- CATCH: log_2.5(new_pokedex_count) per pokedex increment ----
        pokedex_now = rm.pokedex_owned_count(mem)
        if pokedex_now > self._last_pokedex_count:
            for new_count in range(self._last_pokedex_count + 1, pokedex_now + 1):
                if new_count > 1:
                    reward += self._track(
                        "CATCH", math.log(new_count) / math.log(2.5)
                    )
            self._last_pokedex_count = pokedex_now

        # ---- EXPLORE: 0.0001 * N^1.001 per new tile ----
        visited_now = len(self._visited)
        if visited_now > prev_visited_count:
            for n in range(prev_visited_count + 1, visited_now + 1):
                reward += self._track(
                    "EXPLORE", self.EXPLORE_COEF * (n ** self.EXPLORE_EXPONENT)
                )
            self._steps_since_new_tile = 0
        else:
            self._steps_since_new_tile += 1

        # ---- BADGE: 15 * x^1.3 per badge gained ----
        badge_count_now = rm.badges_count(mem)
        if badge_count_now > prev_badge_count:
            for new_count in range(prev_badge_count + 1, badge_count_now + 1):
                reward += self._track(
                    "BADGE", self.BADGE_COEF * (new_count ** self.BADGE_EXPONENT)
                )

        # ---- BEAT_MON: capped 5/area, falling log_2.5(7 - kill_n) ----
        # Detect the same trigger as base BEAT_MON: enemy mon HP just hit 0
        # during an ongoing battle.
        in_battle_now = rm.in_battle(mem)
        if (prev_in_battle != 0 and in_battle_now != 0
                and prev_enemy_hp > 0
                and rm.enemy_mon_hp(mem) == 0):
            current_map = rm.map_id(mem)
            kill_n = self._kills_per_area.get(current_map, 0) + 1
            self._kills_per_area[current_map] = kill_n
            if kill_n <= self.KILL_CAP_PER_AREA:
                reward += self._track(
                    "BEAT_MON", math.log(7 - kill_n) / math.log(2.5)
                )

        # ---- POKEBALL_BUY: 5 * max(0, 1 - n/20) per purchase ----
        pokeballs_now = rm.bag_item_quantity(mem, rm.ITEM_POKEBALL_ID)
        if pokeballs_now > self._last_pokeballs:
            for owned_before in range(self._last_pokeballs, pokeballs_now):
                bonus = self.POKEBALL_BUY_COEF * max(
                    0.0, 1.0 - owned_before / self.POKEBALL_BUY_CAP
                )
                reward += self._track("POKEBALL_BUY", bonus)
        self._last_pokeballs = pokeballs_now

        # ---- STUCK: -0.005/step after 200 steps with no new tile ----
        if self._steps_since_new_tile > self.STUCK_NO_TILE_THRESHOLD:
            reward += self._track("STUCK", self.STUCK_PENALTY_PER_STEP)

        return reward


# Alias: V0.3.4 is the conceptual start of the V0.4 line (full strip +
# curve-based future-seeking design). Kept under the old name for the
# currently-running jobs that loaded the class by that name; future
# configs should reference RewardV0_4_0.
RewardV0_4_0 = RewardV0_3_4


class RewardV0_4_1(RewardV0_4_0):
    """V0.4.1: V0.4.0 + DAMAGE_QUAD penalty symmetric to HEAL_QUAD.

    V0.4.0 (Tail Whip-locked across entropy 0.005 / 0.01 / 0.03):
    rewarded HP gains via the quadratic heal but was silent on HP
    losses. Combined with STEP_PENALTY=0, "stand still in the battle
    menu Tail Whipping" had no per-step cost — even though Squirtle
    was taking damage every turn. The agent learned battles were a
    safe zone and any move that prolonged them was preferable.

    Fix: make damage taken symmetric to healing.
        delta_hp_frac < 0  ->  reward += -(delta)^2 * 15

    The (delta)^2 weighting matches HEAL_QUAD's curve. Per-turn damage
    of ~15% HP pays roughly -0.34. A Tackle-win 4-turn battle taxes
    ~1.4 total. A Tail Whip-stall 10-turn battle taxes ~3.4. Tackle
    becomes mathematically preferable without us adding per-action
    rewards — still pure state-based, still curve-shaped, still V0.4.

    Faint events are excluded (cur==0): FAINT_PENALTY -5 already
    handles them and we don't want to double-bill. Blackout heal
    (HP 0 -> full) is naturally not a damage event so the existing
    blackout guard isn't needed here.
    """

    DAMAGE_QUAD_COEF = 15.0

    def compute(self, mem) -> float:
        # Snapshot prev hp_frac BEFORE super() updates it via heal_quad.
        prev_hp_frac = self._last_hp_frac

        reward = super().compute(mem)

        party_hp = rm.party_hp(mem)
        cur = sum(c for c, _m in party_hp)
        mx = sum(m for _c, m in party_hp)
        if mx > 0 and cur > 0:  # alive — let FAINT_PENALTY handle dead party
            hp_frac = cur / mx
            delta = hp_frac - prev_hp_frac
            if delta < 0 and self.DAMAGE_QUAD_COEF > 0:
                reward += self._track(
                    "DAMAGE_QUAD", -(delta ** 2) * self.DAMAGE_QUAD_COEF
                )

        return reward


class RewardV0_4_2_center(RewardV0_4_1):
    """V0.4.2 (center arm): plug the in-battle menu-stall hole +
    recalibrate negative coefficients that double-counted in V0.4.1.

    Diagnosis from V0.4 baseline + V0.4.1 watch (Day 4-5):
      - V0.4 (no DAMAGE_QUAD): Tail Whip lock-in — in-battle menu free
      - V0.4.1 (DAMAGE_QUAD=15, LOSE=-20): refuses to commit moves at
        all; navigates submenus without committing to avoid enemy turns
        and skip the damage tax. Mean return -19.5 because LOSE_BATTLE
        -20 stacked on DAMAGE_QUAD made bootstrap regime catastrophically
        negative-EV (~-17 at 30% win rate).

    Three-pronged structural fix:

    1. BATTLE_STALL: -0.01/step after 30 consecutive steps of no HP
       change on either side. Mirrors STUCK's activity-based philosophy,
       scoped to in-battle frames. Plugs the menu-stall escape — menu
       navigation that doesn't commit a move no longer evades the
       reward signal entirely. Tackle/Tail Whip don't trigger this
       because enemy turns produce HP delta either way.

    2. DAMAGE_QUAD_COEF: 15 -> 5. Preserves the asymmetry signal
       (Tackle still beats Tail Whip by ~2-3x cumulative damage tax)
       at a magnitude that doesn't dominate bootstrap. Per-turn at
       ~15% HP loss: -0.34 -> -0.11, roughly matches LEVEL increment.

    3. LOSE_BATTLE: -20 -> -10. V0.4's raise to -20 predated
       DAMAGE_QUAD; the latter now handles ongoing damage cost during
       losing fights. -10 keeps blackout meaningfully costly without
       double-counting the same fight's losses.

    Predicted Blue rival EV at 30% win rate: ~-8 (was ~-17 in V0.4.1).
    Breakeven ~57% — within bootstrap-recoverable range.
    """

    # Recalibrated from V0.4.1
    DAMAGE_QUAD_COEF = 5.0
    LOSE_BATTLE = -10.0

    # New BATTLE_STALL machinery
    BATTLE_STALL_THRESHOLD = 30
    BATTLE_STALL_COEF = -0.01

    def __init__(self) -> None:
        super().__init__()
        self._battle_stall_counter = 0

    def reset(self, mem) -> None:
        super().reset(mem)
        self._battle_stall_counter = 0

    def compute(self, mem) -> float:
        # Snapshot pre-state BEFORE super() updates self._last_*
        prev_in_battle = self._last_in_battle
        prev_enemy_hp = self._last_enemy_hp
        prev_party_hp_sum = (
            sum(self._last_party_hp) if self._last_party_hp else None
        )

        reward = super().compute(mem)

        # ---- BATTLE_STALL: penalize sustained no-HP-change in battle ----
        # Counter increments each in-battle step where neither party_hp
        # nor enemy_hp changed since the previous compute(). Resets on
        # any HP delta either side, or on exiting battle.
        in_battle_now = rm.in_battle(mem)
        if in_battle_now != 0 and prev_in_battle != 0:
            party_hp_sum_now = sum(c for c, _m in rm.party_hp(mem))
            enemy_hp_now = rm.enemy_mon_hp(mem)
            hp_changed = (
                (prev_party_hp_sum is not None
                 and party_hp_sum_now != prev_party_hp_sum)
                or (enemy_hp_now != prev_enemy_hp)
            )
            if hp_changed:
                self._battle_stall_counter = 0
            else:
                self._battle_stall_counter += 1
                if self._battle_stall_counter > self.BATTLE_STALL_THRESHOLD:
                    reward += self._track(
                        "BATTLE_STALL", self.BATTLE_STALL_COEF
                    )
        else:
            self._battle_stall_counter = 0

        return reward


class RewardV0_4_2_gentle(RewardV0_4_2_center):
    """V0.4.2 gentle arm: lighter damage signal + lighter loss penalty.
    Bootstrap breakeven ~51%. Tests whether more headroom lets the
    agent commit to combat habits before refining away suboptimal
    moves.
    """
    DAMAGE_QUAD_COEF = 3.0
    LOSE_BATTLE = -7.0


class RewardV0_4_2_harsh(RewardV0_4_2_center):
    """V0.4.2 harsh arm: sharper damage signal + sharper loss penalty.
    Bootstrap breakeven ~64%. Tests whether the in-battle policy will
    commit to Tackle even when losing has real teeth, given that
    BATTLE_STALL has plugged the menu-stall escape.
    """
    DAMAGE_QUAD_COEF = 8.0
    LOSE_BATTLE = -15.0


class RewardV0_3_2_h10(RewardV0_3_1):
    """V0.3.1 + PC_HEAL_LOW bumped 5 -> 10 (conservative arm of the
    h-sweep). Smallest measurable change from V0.3.1; tests whether
    a 2x bump is enough to teach healing without distorting the rest
    of the reward landscape.
    """
    PC_HEAL_LOW = 10.0


class RewardV0_3_2_h25(RewardV0_3_1):
    """V0.3.1 + PC_HEAL_LOW bumped 5 -> 25 (mid-low arm of the
    h-sweep). Heal value matches BEAT_MON (+20) on the same order
    of magnitude; one nurse heal is worth ~one wild kill.
    """
    PC_HEAL_LOW = 25.0


class RewardV0_3_2_h35(RewardV0_3_1):
    """V0.3.1 + PC_HEAL_LOW bumped 5 -> 35 (mid-high arm of the
    h-sweep). Sits at the low end of the math-driven recommended
    range (+35-45) from the EV derivation.
    """
    PC_HEAL_LOW = 35.0


class RewardV0_3_2_h50(RewardV0_3_1):
    """V0.3.1 + PC_HEAL_LOW bumped 5 -> 50 (aggressive arm of the
    h-sweep). 10x bump, in line with the BEAT_MON progression
    (2 -> 10 -> 20). Tests whether overshooting buys cleaner
    discovery of the heal chain at the cost of possible PC loops.
    """
    PC_HEAL_LOW = 50.0


REWARD_REGISTRY: dict[str, type] = {
    "RewardV0_1": RewardV0_1,
    "RewardV0_2": RewardV0_2,
    "RewardV0_2_1": RewardV0_2_1,
    "RewardV0_2_2": RewardV0_2_2,
    "RewardV0_2_3": RewardV0_2_3,
    "RewardV0_2_4_f25": RewardV0_2_4_f25,
    "RewardV0_2_4_f50": RewardV0_2_4_f50,
    "RewardV0_2_4_f100": RewardV0_2_4_f100,
    "RewardV0_2_5_b20": RewardV0_2_5_b20,
    "RewardV0_2_5_b50": RewardV0_2_5_b50,
    "RewardV0_2_5_b100": RewardV0_2_5_b100,
    "RewardV0_2_6": RewardV0_2_6,
    "RewardV0_2_7": RewardV0_2_7,
    "RewardV0_3_1": RewardV0_3_1,
    "RewardV0_3_3": RewardV0_3_3,
    "RewardV0_3_4": RewardV0_3_4,
    "RewardV0_4_0": RewardV0_4_0,
    "RewardV0_4_1": RewardV0_4_1,
    "RewardV0_4_2_center": RewardV0_4_2_center,
    "RewardV0_4_2_gentle": RewardV0_4_2_gentle,
    "RewardV0_4_2_harsh": RewardV0_4_2_harsh,
    "RewardV0_3_2_h10": RewardV0_3_2_h10,
    "RewardV0_3_2_h25": RewardV0_3_2_h25,
    "RewardV0_3_2_h35": RewardV0_3_2_h35,
    "RewardV0_3_2_h50": RewardV0_3_2_h50,
}


def get_reward_cls(name: str) -> type:
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward class {name!r}. "
            f"Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]
