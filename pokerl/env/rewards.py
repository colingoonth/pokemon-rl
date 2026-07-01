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
        self._party_fainted_this_battle = False
        self._last_party_count = 0
        self._visited_maps: set[int] = set()
        self._caught_species: set[int] = set()

    def reset(self, mem: MemoryView) -> None:
        super().reset(mem)
        self._enemy_killed_this_battle = False
        self._pokemon_caught_this_battle = False
        self._battle_was_trainer = False
        self._party_fainted_this_battle = False
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
            self._party_fainted_this_battle = False
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
                # No kill, no catch, party_alive=True. Two cases:
                # 1. Legitimate flee — HP whatever it was, agent ran away
                # 2. Blackout — agent fainted at some point during this
                #    battle and respawned at full HP before battle_ended
                #    was caught. `_party_fainted_this_battle` flag is
                #    sticky across the battle so it survives the gap
                #    between the faint step and the in_battle=0 step.
                if self._party_fainted_this_battle:
                    reward += self.LOSE_BATTLE
                    self._attr_v022("LOSE_BATTLE", self.LOSE_BATTLE)
                else:
                    reward += self.FLEE_PENALTY
                    self._attr_v022("FLEE_PENALTY", self.FLEE_PENALTY)

        # ----- Per-faint penalty -----
        for i in range(min(len(party_hp), len(self._last_party_hp))):
            if self._last_party_hp[i] > 0 and party_hp[i] == 0:
                reward += self.FAINT_PENALTY
                self._attr_v022("FAINT_PENALTY", self.FAINT_PENALTY)
                self._party_fainted_this_battle = True
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
                    self._attr_v022("PC_FIRST_VISIT", self.PC_FIRST_VISIT)
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

        # Single-step blackout signature: battle just ended this step
        # AND HP jumped to full in the same step (delta > 0.5). That's
        # the respawn-at-full-HP pattern after a faint. The PC-map check
        # would be cleaner, but the agent's first blackout sends it to
        # the player's home (Pallet Town bedroom) — not a PC map — until
        # it has visited a Pokemon Center for the first time.
        # Player-initiated PC heals don't pass this check because they
        # happen without a battle->overworld transition this step.
        # In-battle item heals don't pass because in_battle stays nonzero.
        in_battle_now = rm.in_battle(mem)
        single_step_blackout = (
            prev_in_battle != 0 and in_battle_now == 0
            and hp_frac >= 1.0 and delta > 0.5
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
       losing fights. -10 keeps blackout meaningfully costly across
       repeated fights without re-charging the same lost fight's prior
       turns of damage.

       NOTE (whiteout cost is intentionally FAINT-stacked): on a
       whiteout the killing-blow mon also fires FAINT_PENALTY, so a
       single-mon whiteout costs LOSE_BATTLE + FAINT_PENALTY (-15), and
       a multi-mon wipe adds one FAINT_PENALTY per downed mon on top of
       the single -10. This stacking is deliberate (a full wipe should
       hurt more than losing one mon), NOT the double-count this bullet
       warns against — that warning is about not re-billing the same
       fight's damage turns, which DAMAGE_QUAD already covers.

    Predicted Blue rival EV at 30% win rate: ~-8 (was ~-17 in V0.4.1).
    Breakeven ~57% — within bootstrap-recoverable range.
    """

    # Recalibrated from V0.4.1
    DAMAGE_QUAD_COEF = 5.0
    LOSE_BATTLE = -10.0

    # New BATTLE_STALL machinery
    BATTLE_STALL_THRESHOLD = 30
    BATTLE_STALL_COEF = -0.01
    # Per-step penalty ramp: penalty = COEF * min(steps_over_threshold, RAMP).
    # RAMP=1 keeps the original FLAT -0.01/step (current behavior, all classes);
    # a larger RAMP makes sustained stalling escalate then cap (used by V0.5.3+
    # to kill the battle-stall equilibrium).
    BATTLE_STALL_RAMP = 1

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
                over = self._battle_stall_counter - self.BATTLE_STALL_THRESHOLD
                if over > 0:
                    ramp = min(over, self.BATTLE_STALL_RAMP)
                    reward += self._track(
                        "BATTLE_STALL", self.BATTLE_STALL_COEF * ramp
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


class RewardV0_4_3_dense(RewardV0_4_2_center):
    """V0.4.3 (dense story arm): fill the V0.4.2 milestone hole with a
    fine-grained per-event-flag gradient.

    The V0.4 "full strip" zeroed FLAG_REWARD/NEW_MAP_REWARD (rewards.py
    RewardV0_3_4), so V0.4.2 pays nothing for story progress between
    leaving Pallet and the first badge. 50k-step evals of two fresh
    V0.4.2 policies were STUCK 73-82% of steps with no positive attractor
    to climb toward (EXPLORE ~0.04/episode, negligible). This arm restores
    the missing attractor as STATE-based outcome supervision (consistent
    with the V0.4 philosophy): +FLAG_BIT_REWARD the first time each
    individual event-flag bit transitions 0->1 during an episode.

    Anti-farm: each of the ~2560 bit-indices is paid at most once per
    episode (`_rewarded_bits` mask, updated unconditionally so a bit that
    sets / clears / re-sets is never paid twice), and total flag reward is
    hard-capped per episode (FLAG_REWARD_EPISODE_CAP) so it stays
    structurally sub-badge and cannot dominate combat/survival.

    STUCK and entropy are deliberately UNCHANGED from V0.4.2 — this arm is
    the clean A/B control against RewardV0_4_3_curated, which differs only
    in the story-reward mechanism.
    """

    FLAG_BIT_REWARD = 1.0
    FLAG_REWARD_EPISODE_CAP = 40.0

    def __init__(self) -> None:
        super().__init__()
        self._rewarded_bits: set[int] = set()
        self._flag_reward_accum = 0.0

    def reset(self, mem) -> None:
        super().reset(mem)
        # Pre-mask every flag already set at episode start (blue_fight.state
        # has the rival battle done) so we only pay for in-episode progress.
        self._rewarded_bits = rm.event_flag_bits_set(mem)
        self._flag_reward_accum = 0.0

    def compute(self, mem) -> float:
        reward = super().compute(mem)

        now = rm.event_flag_bits_set(mem)
        new_bits = now - self._rewarded_bits
        if new_bits and self._flag_reward_accum < self.FLAG_REWARD_EPISODE_CAP:
            headroom = self.FLAG_REWARD_EPISODE_CAP - self._flag_reward_accum
            pay = min(len(new_bits) * self.FLAG_BIT_REWARD, headroom)
            self._flag_reward_accum += pay
            reward += self._track("FLAG_BIT", pay)
        # Mask unconditionally (even past the cap) so post-cap toggles can't
        # be farmed on a later step.
        self._rewarded_bits |= new_bits

        return reward


class RewardV0_4_3_curated(RewardV0_4_2_center):
    """V0.4.3 (curated story arm): fill the V0.4.2 milestone hole with
    chunky map-progression goals instead of per-flag signal.

    Same diagnosis as RewardV0_4_3_dense, opposite mechanism: rather than
    rewarding all ~2560 event bits, reward a small curated set of map-id
    milestones (one-shot on first entry) plus a per-new-map bonus plus a
    coarse popcount-delta flag proxy as intermediate gradient. Named
    per-flag milestones (got-parcel, got-pokedex) are not used because
    ram_map has no verified per-bit reader — map-ids are fully supported.

    NOTE: NEW-map signal is implemented LOCALLY here (NEW_MAP_BONUS + own
    `_seen_maps`) rather than by re-enabling the ancestor's NEW_MAP_REWARD
    constant. The V0.2.2 NEW_MAP path still runs through super().compute()
    and would double-count (and land in UNATTRIBUTED) if that constant were
    set; keeping it at 0 and tracking our own keeps attribution clean.

    STUCK and entropy are deliberately UNCHANGED from V0.4.2 (clean A/B
    control vs RewardV0_4_3_dense).
    """

    NEW_MAP_BONUS = 3.0
    # One-shot bonus on first entry to each curated story map (verified ids).
    MILESTONE_MAP_BONUS = {
        rm.MAP_ROUTE_1: 4.0,       # 0x0C — left Pallet, on the road
        rm.MAP_VIRIDIAN_CITY: 6.0,  # 0x01 — reached the first city
        rm.MAP_PEWTER_CITY: 10.0,  # 0x02 — through the forest, Brock's town
        rm.MAP_PEWTER_GYM: 15.0,   # 0x36 — entered Brock's gym (V1 target)
    }
    FLAG_POPCOUNT_BONUS = 1.0      # per net-new event flag (high-water mark)

    def __init__(self) -> None:
        super().__init__()
        self._seen_maps: set[int] = set()
        self._fired_map_milestones: set[int] = set()
        self._milestone_base_flagcount = 0

    def reset(self, mem) -> None:
        super().reset(mem)
        self._seen_maps = {rm.map_id(mem)}
        self._fired_map_milestones = set()
        # Only pay for flags set AFTER the start state.
        self._milestone_base_flagcount = rm.event_flags_popcount(mem)

    def compute(self, mem) -> float:
        reward = super().compute(mem)

        cur_map = rm.map_id(mem)

        # ---- NEW_MAP: per previously-unseen map_id (one-shot) ----
        if cur_map not in self._seen_maps:
            self._seen_maps.add(cur_map)
            reward += self._track("NEW_MAP", self.NEW_MAP_BONUS)

        # ---- MILESTONE_MAP: curated story maps (one-shot) ----
        if (cur_map in self.MILESTONE_MAP_BONUS
                and cur_map not in self._fired_map_milestones):
            self._fired_map_milestones.add(cur_map)
            reward += self._track(
                "MILESTONE_MAP", self.MILESTONE_MAP_BONUS[cur_map]
            )

        # ---- FLAG_PROGRESS: +1 per net-new event flag (high-water mark) ----
        flags_now = rm.event_flags_popcount(mem)
        if flags_now > self._milestone_base_flagcount:
            gained = flags_now - self._milestone_base_flagcount
            reward += self._track(
                "FLAG_PROGRESS", self.FLAG_POPCOUNT_BONUS * gained
            )
            self._milestone_base_flagcount = flags_now

        return reward


class RewardV0_4_4_dense(RewardV0_4_3_dense):
    """V0.4.4 dense: V0.4.3 dense per-flag story reward + restored
    PC_FIRST_VISIT (+10, one-shot per Pokecenter).

    Fixes the heal-gate found 2026-06-10: both V0.4.3 arms stuck in a low-HP
    forced-flee loop (single Pokemon, hurt on Route 1, can't switch, won't
    path to the nurse). HEAL_QUAD only pays AFTER a heal completes, giving no
    gradient toward the Center. This re-enables the V0.2.3 PC machinery
    (entry detection, _visited_pokecenters, day-7 blackout suppression) that
    the V0.4 strip zeroed (RewardV0_3_4 line ~878) — only the constant
    changes, per the project discipline. HEAL_QUAD then pays the (near-full,
    since it arrives low) heal on top. Tests whether unblocking healing lets
    the dense story signal drive forward progress.
    """
    PC_FIRST_VISIT = 10.0


class RewardV0_4_4_curated(RewardV0_4_3_curated):
    """V0.4.4 curated: V0.4.3 curated map-milestone story reward + restored
    PC_FIRST_VISIT (+10, one-shot per Pokecenter). A/B partner to
    RewardV0_4_4_dense — differs ONLY in the story-reward mechanism; the heal
    fix is identical. See RewardV0_4_4_dense for the heal-gate rationale.
    """
    PC_FIRST_VISIT = 10.0


class RewardV0_4_5_dense(RewardV0_4_4_dense):
    """V0.4.5 dense: V0.4.4 + PC_FIRST_VISIT bumped 10 -> 50.

    Magnitude test of the heal-entry incentive. At +10 (V0.4.4) the agent
    reached a Pokecenter ~once per 30k steps and stood in Viridian ignoring
    the Center — EXPLORE/NEW_MAP pay it to keep walking past. +50 makes
    entering clearly worth the detour. One-shot per Center per episode (no
    farm) and bounded (~1-2 reachable Centers early), so it stays on the
    critical path rather than dominating. Intended for WARM-START from a
    V0.4.4 iter-~2800 checkpoint (already explores + reaches Viridian), so
    we test door-entry behavior in minutes, not hours. entropy_coef 0.01.
    """
    PC_FIRST_VISIT = 50.0


class RewardV0_4_5_curated(RewardV0_4_4_curated):
    """V0.4.5 curated: V0.4.4 + PC_FIRST_VISIT bumped 10 -> 50. A/B partner
    to RewardV0_4_5_dense; warm-started from V0.4.4 curated iter-~2800.
    """
    PC_FIRST_VISIT = 50.0


class RewardV0_5_skeleton(RewardV0_4_2_center):
    """V0.5: thin extrinsic skeleton for the RND / curiosity build.

    The V0.5 pivot moves *exploration* off hand-engineered tile rewards and
    onto an intrinsic curiosity signal (RND — computed in the PPO loop, not
    here). So this reward keeps only the parts that are NOT exploration
    shaping, and adds the true story gates as one-shot objective bonuses:

      KEEP (inherited from V0.4.2_center):
        - combat / survival: HEAL_QUAD, DAMAGE_QUAD, BATTLE_STALL, FAINT,
          LOSE_BATTLE, FLEE. The agent still has to fight to reach Brock;
          this layer was proven across V0.4.1/4.2 to kill the Tail-Whip
          menu-stall. It is NOT path force-feeding — it teaches *fighting*.
        - progression: LEVEL (log_6), BADGE (power-law). The "what."

      ADD:
        - three one-shot HARD-GATE bonuses on the forced path to Brock:
          got Oak's Parcel -> got Pokedex (the forced Pallet backtrack) ->
          beat Brock. Objectives, not paths. Curiosity finds *how*; these
          mark *that* the milestone is worth something.

      REMOVE vs V0.4.2:
        - EXPLORE curve (EXPLORE_COEF -> 0): replaced by RND novelty. This
          was the coordinate-exploration reward that "wanders forever, never
          reaches Brock"; intrinsic curiosity replaces it.
        - STUCK anti-camp penalty (-> 0): RND's decaying novelty handles
          camping on its own, and the explicit penalty also punished the
          legitimate south-bound parcel backtrack.

    Gate bonuses fire exactly once per episode (tracked in _gates_paid),
    are pre-masked in reset() so a warm-started / mid-game start state does
    not re-pay past progress, and attribute via _track so they appear in the
    watcher breakdown rather than UNATTRIBUTED.
    """

    EXPLORE_COEF = 0.0             # exploration is now RND's job
    STUCK_PENALTY_PER_STEP = 0.0   # drop anti-camp penalty (it punished the backtrack)

    # One-shot hard-gate bonuses (true objectives on the path to Brock).
    # Tunable; chosen meaningfully above the per-step noise floor but rare
    # (one-shot) so they don't dominate the dense combat/level signal.
    GATE_GOT_OAKS_PARCEL = 10.0
    GATE_GOT_POKEDEX     = 25.0    # the forced Pallet backtrack — hardest gate
    GATE_BEAT_BROCK      = 50.0    # V1 goal

    def __init__(self) -> None:
        super().__init__()
        self._gates_paid: set[str] = set()

    def reset(self, mem) -> None:
        super().reset(mem)
        self._gates_paid = set()
        # Pre-mask gates already satisfied at the start state so a warm-start
        # or mid-game curriculum state doesn't pay for past progress.
        if rm.got_oaks_parcel(mem):
            self._gates_paid.add("GATE_GOT_OAKS_PARCEL")
        if rm.got_pokedex(mem):
            self._gates_paid.add("GATE_GOT_POKEDEX")
        if rm.beat_brock(mem):
            self._gates_paid.add("GATE_BEAT_BROCK")

    def compute(self, mem) -> float:
        reward = super().compute(mem)
        for name, fired, amount in (
            ("GATE_GOT_OAKS_PARCEL", rm.got_oaks_parcel(mem), self.GATE_GOT_OAKS_PARCEL),
            ("GATE_GOT_POKEDEX",     rm.got_pokedex(mem),     self.GATE_GOT_POKEDEX),
            ("GATE_BEAT_BROCK",      rm.beat_brock(mem),      self.GATE_BEAT_BROCK),
        ):
            if fired and name not in self._gates_paid:
                self._gates_paid.add(name)
                reward += self._track(name, amount)
        return reward


class RewardV0_5_1_storyladder(RewardV0_4_2_center):
    """V0.5.1: the V0.5 RND combat skeleton + a story-progress MULTIPLIER ladder.

    Diagnosis (Day-11): RAM-feature RND (job 22995) broke the exploration
    plateau (tiles/env 1 -> 661) but the agent earned its ENTIRE return from
    combat (HEAL_QUAD, BEAT_MON, LEVEL) and ZERO from story — it covers the map
    but never plays the game. Two structural causes:
      1. The only extrinsic story signal was three sparse one-shot gates, so
         there is no gradient pulling the policy ALONG the story.
      2. The forced Oak backtrack (parcel -> Pokedex) is HARD-GATED: the
         Viridian old man (ViridianCity.asm) physically shoves the player south
         off the north exit until EVENT_GOT_POKEDEX is set. The agent literally
         cannot reach Viridian Forest without completing the parcel loop, and
         nothing told it to.

    Design (your call, validated by research): instead of additive constants,
    a story multiplier `M(stage)` that scales the dense combat/level/heal
    signal, so a high score REQUIRES story progress (grinding alone caps out).
    Each verified, monotone, on-path checkpoint is a "rung": reaching it (a) adds
    a one-shot bonus (the dense pull + a value-seed past PPO's advantage
    normalization, which would otherwise wash out a pure scale) and (b) raises
    M for all subsequent reward. M climbs ~2.85x at the forest, ~4.35x at Brock.

    Honest scope (Day-11 3-researcher audit): verified monotone flags are SPARSE
    — there are three flag DESERTS (Route 1, the parcel backtrack, the Forest
    maze) with no on-path flag. This ladder does NOT pretend to flag-guide those:
    map-entry rungs break them up, and RND carries the rest — in particular the
    parcel-bit flip re-spikes RND novelty across the backtrack tiles (the
    mechanism RAM-RND was built for). Flags+maps are the anchors; RND is the
    transport between them. Avoidable/off-path/non-monotone flags (Forest
    trainers, museum, BOUGHT_MUSEUM_TICKET which Pewter's script resets each
    frame) are deliberately excluded.

    Inherits RewardV0_4_2_center (combat/level/heal) directly and re-zeroes
    EXPLORE/STUCK like RewardV0_5_skeleton; the skeleton's 3 gates are subsumed
    into the ladder below (parcel/pokedex/brock), so this does NOT extend the
    skeleton (would double-pay).
    """

    EXPLORE_COEF = 0.0             # exploration is RND's job
    STUCK_PENALTY_PER_STEP = 0.0   # anti-camp penalty punished the backtrack

    # Ordered story ladder, start-state -> Brock. Each entry:
    #   (key, kind, target, one_shot_bonus, mult_increment)
    # kind "flag": target is an rm.<predicate>(mem)->bool (verified flag bit).
    # kind "map":  target is a map_id; the rung fires on FIRST visit to that map.
    # All flags/maps cross-checked vs pret/pokered + pokemonred_puffer (Day-11).
    _LADDER: "list[tuple]" = [
        ("POKEBALLS",     "flag", rm.got_pokeballs_from_oak,           5.0,  0.10),
        ("ROUTE_1",       "map",  rm.MAP_ROUTE_1,                      4.0,  0.10),
        ("POTION_SAMPLE", "flag", rm.got_potion_sample,               3.0,  0.05),
        ("VIRIDIAN",      "map",  rm.MAP_VIRIDIAN_CITY,                6.0,  0.15),
        ("VIRIDIAN_MART", "map",  rm.MAP_VIRIDIAN_MART,                4.0,  0.05),
        ("PARCEL",        "flag", rm.got_oaks_parcel,                 10.0,  0.20),
        # --- backtrack staircase: parcel -> Oak (the flagless desert RND alone
        #     never crossed in a full 30M run). Progress-conditioned re-entry. ---
        ("RETURN_ROUTE_1", "map_flag", (rm.MAP_ROUTE_1, rm.got_oaks_parcel),     6.0,  0.10),  # halfway home
        ("RETURN_PALLET",  "map_flag", (rm.MAP_PALLET_TOWN, rm.got_oaks_parcel),  8.0,  0.15),  # back in town
        ("AT_OAK_PARCEL",  "map_flag", (rm.MAP_OAKS_LAB, rm.got_oaks_parcel),    12.0,  0.20),  # at the delivery point
        ("POKEDEX",       "flag", rm.got_pokedex,                     30.0,  0.50),  # ★ opens north gate
        ("ROUTE_2",       "map",  rm.MAP_ROUTE_2,                     12.0,  0.30),  # north is now passable
        ("FOREST_SOUTH",  "map",  rm.MAP_VIRIDIAN_FOREST_SOUTH_GATE,  20.0,  0.40),  # ◆ frontier goal
        ("FOREST",        "map",  rm.MAP_VIRIDIAN_FOREST,             4.0,  0.10),
        ("FOREST_NORTH",  "map",  rm.MAP_VIRIDIAN_FOREST_NORTH_GATE,   8.0,  0.20),
        ("PEWTER",        "map",  rm.MAP_PEWTER_CITY,                 15.0,  0.30),
        ("PEWTER_GYM",    "map",  rm.MAP_PEWTER_GYM,                  15.0,  0.30),
        ("TM34",          "flag", rm.got_tm34,                        10.0,  0.10),
        ("BEAT_BROCK",    "flag", rm.beat_brock,                      50.0,  0.50),  # ★ V1 goal
    ]

    def __init__(self) -> None:
        super().__init__()
        self._fired: set[str] = set()
        self._mult = 1.0
        self._seen_maps: set[int] = set()

    def _rung_hit(self, mem, kind, target) -> bool:
        if kind == "flag":
            return target(mem)
        if kind == "map":
            return target in self._seen_maps
        # "map_flag": progress-conditioned re-entry — fires only while physically
        # ON `map_id` AND a story predicate holds. Densifies the parcel->Oak
        # BACKTRACK, a flagless desert that plain first-entry map rungs miss
        # (Route 1 / Pallet / Oak's Lab were all visited outbound, so re-entry
        # carrying the parcel is the only new monotone state on the return leg).
        map_id, predicate = target
        return rm.map_id(mem) == map_id and predicate(mem)

    def reset(self, mem) -> None:
        super().reset(mem)
        self._fired = set()
        self._mult = 1.0
        # Pre-mask rungs already satisfied at the (warm/mid-game) start state so
        # we don't pay one-shots for past progress, but DO seed the multiplier
        # to the correct starting stage.
        self._seen_maps = {rm.map_id(mem)}
        for key, kind, target, _bonus, inc in self._LADDER:
            if self._rung_hit(mem, kind, target):
                self._fired.add(key)
                self._mult += inc

    def compute(self, mem) -> float:
        base = super().compute(mem)
        self._seen_maps.add(rm.map_id(mem))

        # Scale the dense combat/level/heal signal by the story multiplier.
        # Tracked as its own component (M-1)*base so the attribution breakdown
        # still sums to the returned reward.
        reward = base + self._track("STORY_MULT", (self._mult - 1.0) * base)

        # Fire any newly-reached rungs: one-shot bonus + raise M for next step.
        for key, kind, target, bonus, inc in self._LADDER:
            if key in self._fired:
                continue
            if self._rung_hit(mem, kind, target):
                self._fired.add(key)
                self._mult += inc
                reward += self._track("RUNG_" + key, bonus)
        return reward


class RewardV0_5_2_storyladder(RewardV0_5_1_storyladder):
    """V0.5.2: storyladder + potential-based "danger-zone" HP, REPLACING the
    farmable quadratic heal/damage.

    The V0.5.1 probe (job 23408) reproduced the heal-farm. HEAL_QUAD is a
    quadratic on the heal TRANSITION; because PC heals are instantaneous (one
    big delta) while battle damage is gradual (many small deltas), the quadratic
    always rewards the big heal more than it penalizes the slow damage that set
    it up -> net-positive self-harm farm (+22.7 heal vs -1.5 damage in eval),
    with the agent stalling battles to bleed HP for big heals (827 stall fires).

    Fix (Colin's framing: "healing is a necessary rest after a fight, not a
    reward to farm"): reward the HP STATE via a potential, not the heal act.
        phi(hp_frac) = -C * max(0, T - hp_frac)**2          # T = 0.5, C = 8
        r += phi(hp_now) - phi(hp_prev)
    Properties:
      - Potential-based -> telescopes -> granularity-invariant, so the
        instant-heal-vs-gradual-damage exploit vanishes; a hurt->heal round
        trip nets ~zero. Healing repays the debt; it is not profit.
      - Flat above T = 50% HP: a WON fight that ends healthy costs NOTHING, so
        the agent is never punished for winning -> no fight-timidity (the
        explicit risk). The term only bites in the 0-50% danger zone, leaning
        the agent to rest before walking into the next fight hurt.
      - C = 8 pins the worst case (full plunge 0.5->0) at -0.25*C = -2.0, ~one
        kill -> HP is minor upkeep, dominated by combat + the story multiplier
        + FAINT(-5)/LOSE(-10). Sizing from two independent analyses (window
        [8,16]); chose the timidity-safe floor. Bump toward 12 if the probe
        shows the agent ignoring HP and fainting.

    HEAL_QUAD / DAMAGE_QUAD are DISABLED (coef 0) — the danger-zone term
    REPLACES them. Added unmultiplied (after the story multiplier) so the
    potential stays policy-invariant (a per-step M would break telescoping).
    The faint->respawn HP jump (prev == 0 -> full) is skipped so respawning
    cannot refund the faint.
    """

    HEAL_QUAD_COEF = 0.0       # disabled — replaced by potential-based danger-zone HP
    DAMAGE_QUAD_COEF = 0.0     # disabled — same
    HP_DANGER_C = 8.0          # potential scale; worst-case debt = 0.25*C = 2.0 (~one kill)
    HP_DANGER_T = 0.5          # danger threshold: term is flat (0) above this HP fraction

    def _hp_potential(self, f: float) -> float:
        gap = self.HP_DANGER_T - f
        return -self.HP_DANGER_C * gap * gap if gap > 0.0 else 0.0

    def compute(self, mem) -> float:
        prev_hp = self._last_hp_frac          # end-of-previous-step HP fraction
        reward = super().compute(mem)         # storyladder (mult + rungs); heal/dmg quads off
        now_hp = self._last_hp_frac           # super chain updated this to current HP
        # Skip the faint->respawn jump: prev == 0 means the party was fainted,
        # so any rise from 0 is a blackout respawn, not a chosen heal.
        if prev_hp > 1e-6:
            dphi = self._hp_potential(now_hp) - self._hp_potential(prev_hp)
            if dphi != 0.0:
                reward += self._track("HP_DANGER", dphi)
        return reward


class RewardV0_5_3_storyladder(RewardV0_5_2_storyladder):
    """V0.5.3: fix the battle-stall equilibrium that V0.5.2 exposed.

    The V0.5.2 probe (job 23421) collapsed onto a degenerate stall: the argmax
    policy sat in the OPENING rival battle for 32715/32768 steps, never
    resolving it (0 rungs, 0 tiles). Removing the heal-farm removed the agent's
    (crude) reason to fight; with fighting all-downside (faint/lose/HP_DANGER)
    and stalling costing a flat -0.01/step, stalling became the safest action.

    Two changes, no new farm:
      1. RIVAL_WIN rung (+25, first on the ladder): a big one-shot for winning
         the opening rival battle (EVENT_BATTLED_RIVAL_IN_OAKS_LAB). The rival
         fight is where fight-reluctance is WORST — longest early battle, ends
         lowest HP, so HP_DANGER deters it most — and it is exactly where the
         agent froze. A large win bonus offsets that deterrent where it bites
         hardest. One-shot flag -> non-farmable. The rival battle is a TRAINER
         battle (in_battle == 2, unfleeable), so with (2) the agent's only good
         move is to attack and win.
      2. Escalating BATTLE_STALL (BATTLE_STALL_RAMP = 100): the stall penalty
         now GROWS with sustained stalling (-0.01/step ramping to -1.0/step over
         ~100 over-threshold steps, then capped) instead of a flat -0.01.
         Indefinite stalling becomes catastrophic in ANY battle, so the stall
         can't simply relocate to a later fight. (Base V0.4.2 keeps RAMP = 1 =
         the old flat behavior.)
    """

    BATTLE_STALL_RAMP = 100   # escalate stall penalty (flat=1): -0.01 -> -1.0/step cap

    # RIVAL_WIN prepended as the opening beat; the rest of the ladder inherited.
    _LADDER = [
        ("RIVAL_WIN", "flag", rm.battled_rival_in_oaks_lab, 25.0, 0.15),
    ] + RewardV0_5_1_storyladder._LADDER


class RewardV0_5_4_storyladder(RewardV0_5_3_storyladder):
    """V0.5.4: dense enemy-damage "engagement" reward — the function the
    heal-farm was secretly providing.

    Three probes converged on this. WITH the heal-farm (v0.5.1) the agent won
    the rival and reached Route 1; WITHOUT it (v0.5.2/.3) it froze in the FIGHT
    menu and stalled the opening battle for the whole episode, even when
    stalling was made catastrophic (-32688) AND winning paid +25. So incentives
    were never the bottleneck — the agent had no DENSE gradient for the
    fine-grained "land a hit" behavior. The heal-farm accidentally supplied it
    (it rewarded the HP swings of trading blows); removing it left only
    end-of-battle (win) and absence-of-progress (stall) signals, neither of
    which teaches attacking.

    Fix: reward dealing damage directly. ENEMY_DMG = COEF * (enemy HP knocked
    off this step), LINEAR so it's granularity-invariant (no farm-by-chunking)
    and bounded per battle by the enemy's HP. Only positive deltas count, so a
    fresh enemy being sent in (HP jumps up) or battle start pays nothing. This
    is the clean, non-farmable version of the heal-farm's hidden role: a per-hit
    gradient that pulls the policy out of the menu-stall toward winning.
    RIVAL_WIN (+25) stays as the capstone.

    Also reverts the escalating stall to FLAT (BATTLE_STALL_RAMP = 1): the
    v0.5.3 ramp to -1.0/step produced -32688 episodes whose variance drowned the
    learning signal under PPO's per-batch advantage normalization. With a real
    positive reason to attack, the gentle flat -0.01 stall (which sufficed in
    v0.5.1) is enough.
    """

    BATTLE_STALL_RAMP = 1          # revert the v0.5.3 variance-bomb ramp to flat
    ENEMY_DMG_COEF = 0.15          # reward per point of enemy HP dealt (dense engagement)

    def compute(self, mem) -> float:
        prev_enemy_hp = self._last_enemy_hp
        prev_in_battle = self._last_in_battle
        reward = super().compute(mem)        # V0.5.3 chain updates _last_enemy_hp
        # Dense engagement signal: reward enemy HP knocked off this step.
        if rm.in_battle(mem) != 0 and prev_in_battle != 0:
            dmg = prev_enemy_hp - rm.enemy_mon_hp(mem)
            if dmg > 0:
                reward += self._track("ENEMY_DMG", self._enemy_dmg_amount(dmg, mem))
        return reward

    def _enemy_dmg_amount(self, dmg: float, mem) -> float:
        """Engagement reward for `dmg` enemy HP knocked off this step. V0.5.4 is
        flat linear (granularity-invariant, bounded per battle by enemy HP).
        Subclasses override to reshape it — V0.5.5 makes it per-area diminishing
        so grinding one area decays while fresh territory pays full."""
        return self.ENEMY_DMG_COEF * dmg


class RewardV0_5_5_storyladder(RewardV0_5_4_storyladder):
    """V0.5.5: stabilize the progress corridor.

    The v0.5.4 probe (job 23438) reached Oak's Parcel (5 rungs, M=1.65x) but
    could NOT HOLD it — the policy oscillated violently between two attractors
    and never retained progress past Route 1:
      - GRIND: v0.5.4's ENEMY_DMG is flat linear and UNBOUNDED per area (unlike
        BEAT_MON, which is capped 5/area). Amplified by the rising story
        multiplier, farming wild battles in one area paid unboundedly, so the
        policy kept stalling to grind instead of progressing.
      - FLEE-LOOP: fleeing is the only way to cross grass (Gen-1 trainer battles
        are unfleeable, so EVERY flee is a wild traversal-flee), yet FLEE_PENALTY
        taxed each one. An agent trying to traverse got trapped racking up the
        penalty (357 fires / -178 in one eval), the recurring flee-everything
        mode documented back in V0.2.5-0.2.7.

    The corridor between them — "fight through a fresh area, then move on" — was
    too narrow to stay on, so on entropy anneal the policy fell into one
    attractor or the other. Both failure modes share ONE root: nothing made
    *forward* engagement worth more than *repeated* engagement in the same spot.

    Fix (one coupled mechanism, per-area engagement budget):
      A. ENEMY_DMG per-area DIMINISHING. Track cumulative damage dealt per map;
         scale the reward by `max(0, 1 - spent/BUDGET)`. The first ~BUDGET HP of
         damage in a FRESH area pays ~full engagement; grinding the same area
         decays to zero; walking into a new area resets to full. Kills the grind
         attractor — farming Route 1 stops paying, progressing to new territory
         pays. Mirrors the proven BEAT_MON per-area decay, on continuous damage.
      B. COUPLED flee relief. Once an area's engagement budget is spent (area
         "cleared"), fleeing its wild encounters is FREE. Early in a fresh area
         fighting is still incentivized (flee penalized, ENEMY_DMG full); once
         you've engaged enough, the trap is removed and you traverse freely. One
         shared per-area counter drives both A and B, so the optimal policy is
         exactly "engage a new area, then walk through it" — the corridor.

    No new exploit: after the budget is spent, ENEMY_DMG pays 0 AND flee pays 0,
    so lingering earns nothing; the story multiplier's per-step pressure + the
    unreached rungs pull the policy forward.

    BUDGET = 100 HP ~= 5-8 early wild mons, matching BEAT_MON's 5-kill/area cap
    in spirit. Tunable — the one science constant introduced here.
    """

    ENEMY_DMG_BUDGET_PER_AREA = 100.0   # HP of damage per map before engagement saturates

    def reset(self, mem) -> None:
        super().reset(mem)
        self._enemy_dmg_per_area: dict[int, float] = {}

    def _enemy_dmg_amount(self, dmg: float, mem) -> float:
        # Per-area diminishing: full coef while the area's budget is unspent,
        # decaying linearly to 0 as cumulative damage there reaches BUDGET.
        current_map = rm.map_id(mem)
        spent = self._enemy_dmg_per_area.get(current_map, 0.0)
        frac_remaining = max(0.0, 1.0 - spent / self.ENEMY_DMG_BUDGET_PER_AREA)
        self._enemy_dmg_per_area[current_map] = spent + dmg
        return self.ENEMY_DMG_COEF * dmg * frac_remaining

    def compute(self, mem) -> float:
        reward = super().compute(mem)
        # Coupled flee relief: a flee fired this step iff super() tracked a
        # FLEE_PENALTY (last_components is per-step). If the current area's
        # engagement budget is already spent, the area is "cleared" — refund the
        # penalty so traversal is free and the agent isn't trapped.
        flee_pen = self.last_components.get("FLEE_PENALTY", 0.0)
        if flee_pen < 0.0:
            current_map = rm.map_id(mem)
            if self._enemy_dmg_per_area.get(current_map, 0.0) >= self.ENEMY_DMG_BUDGET_PER_AREA:
                reward -= flee_pen                       # flee_pen<0 → add it back
                self.last_components.pop("FLEE_PENALTY", None)
        return reward


# --- V0.5.7 directional field: static critical-path map graph + BFS distances ---
# Hand-authored connectivity of the ~11 maps on the forced path to Brock (verified
# vs pret/pokered map warps). Game-STRUCTURE constant (not RAM state), so it is a
# fixed dict, BFS'd once at import into distance-to-goal tables. The field is then
# a pure dict lookup per step — zero graph work in the hot loop.
_FIELD_GRAPH: "dict[int, list[int]]" = {
    rm.MAP_PALLET_TOWN:                [rm.MAP_ROUTE_1, rm.MAP_OAKS_LAB],
    rm.MAP_ROUTE_1:                    [rm.MAP_PALLET_TOWN, rm.MAP_VIRIDIAN_CITY],
    rm.MAP_VIRIDIAN_CITY:              [rm.MAP_ROUTE_1, rm.MAP_VIRIDIAN_MART, rm.MAP_ROUTE_2],
    rm.MAP_OAKS_LAB:                   [rm.MAP_PALLET_TOWN],
    rm.MAP_VIRIDIAN_MART:              [rm.MAP_VIRIDIAN_CITY],
    rm.MAP_ROUTE_2:                    [rm.MAP_VIRIDIAN_CITY, rm.MAP_VIRIDIAN_FOREST_SOUTH_GATE],
    rm.MAP_VIRIDIAN_FOREST_SOUTH_GATE: [rm.MAP_ROUTE_2, rm.MAP_VIRIDIAN_FOREST],
    rm.MAP_VIRIDIAN_FOREST:            [rm.MAP_VIRIDIAN_FOREST_SOUTH_GATE, rm.MAP_VIRIDIAN_FOREST_NORTH_GATE],
    rm.MAP_VIRIDIAN_FOREST_NORTH_GATE: [rm.MAP_VIRIDIAN_FOREST, rm.MAP_PEWTER_CITY],
    rm.MAP_PEWTER_CITY:                [rm.MAP_VIRIDIAN_FOREST_NORTH_GATE, rm.MAP_PEWTER_GYM],
    rm.MAP_PEWTER_GYM:                 [rm.MAP_PEWTER_CITY],
}


def _bfs_dist(graph: "dict[int, list[int]]", goal: int) -> "dict[int, int]":
    """Hop-count from every reachable map to `goal` over the undirected graph."""
    from collections import deque
    dist = {goal: 0}
    q = deque([goal])
    while q:
        m = q.popleft()
        for nb in graph.get(m, ()):
            if nb not in dist:
                dist[nb] = dist[m] + 1
                q.append(nb)
    return dist


# One distance table per objective map: Viridian Mart (get parcel), Oak's Lab
# (deliver parcel — the backtrack), Pewter Gym (post-Pokedex goal).
_FIELD_DIST: "dict[int, dict[int, int]]" = {
    g: _bfs_dist(_FIELD_GRAPH, g)
    for g in (rm.MAP_VIRIDIAN_MART, rm.MAP_OAKS_LAB, rm.MAP_PEWTER_GYM)
}


class RewardV0_5_7_fieldcount(RewardV0_5_5_storyladder):
    """V0.5.7: V0.5.5 + a directional POTENTIAL FIELD, paired with count-based
    novelty at a REDUCED int_coef (set in the config, not here).

    The V0.5.6 count-novelty probe (job 51276) revived exploration (novelty ~0.14,
    non-saturating, up to 119k tiles/rollout) but REGRESSED the warm-started 3-rung
    policy to 1-2 rungs: undirected coverage out-competed directed story progress
    and diffused the Viridian commitment, so the agent never reached the parcel and
    the reactivation mechanism was never even exercised. Lesson (the design fleet's
    central thesis): curiosity TRANSPORTS but does not DIRECT — it needs an anchor.

    This adds A's telescoping distance-to-objective field on the EXTRINSIC stream,
    un-multiplied (added after the story multiplier, like HP_DANGER, so it
    telescopes and stays policy-invariant / non-farmable). Count-novelty
    (rnd_input='count') stays as the within-map transport but at a subordinate
    int_coef so it assists coverage without swamping the directional pull.

        objective g(s):  Viridian Mart   if not parcel and not pokedex   (get parcel)
                         Oak's Lab        if parcel and not pokedex       (THE backtrack)
                         Pewter Gym       if pokedex and not brock        (north open)
                         field off        if brock

        r_field = W * (d_prev - d_now)      # = phi(s')-phi(s), phi = -W*d, gamma=1

    Non-farmable: the per-episode sum telescopes to W*(d_0 - d_T), so any closed
    walk nets exactly 0 — no loop farm; standing still pays 0 (no dawdle). The
    field RE-BASELINES (pays 0 that step) on an objective flip or an off-graph
    step, so there is no spurious jump and no double-count with the rung one-shots.
    W=2: one map-hop = +2, ~1/10 of a BEAT_MON, deliberately below combat.
    """

    W_FIELD = 2.0

    def _field_objective(self, mem):
        if rm.beat_brock(mem):        return None
        if rm.got_pokedex(mem):       return rm.MAP_PEWTER_GYM
        if rm.got_oaks_parcel(mem):   return rm.MAP_OAKS_LAB
        return rm.MAP_VIRIDIAN_MART

    def _field_dist_now(self, mem):
        goal = self._field_objective(mem)
        m = rm.map_id(mem)
        if goal is None or m not in _FIELD_DIST[goal]:
            return None, goal                       # off-graph / done: field paused
        return _FIELD_DIST[goal][m], goal

    def reset(self, mem) -> None:
        super().reset(mem)
        self._field_prev_d, self._field_prev_goal = self._field_dist_now(mem)

    def compute(self, mem) -> float:
        reward = super().compute(mem)
        d_now, goal = self._field_dist_now(mem)
        d_prev, prev_goal = self._field_prev_d, self._field_prev_goal
        # Emit only when the objective is unchanged AND both endpoints on-graph;
        # otherwise pay 0 and re-baseline (flag-flip jump / off-graph excursion).
        if goal == prev_goal and d_now is not None and d_prev is not None:
            df = self.W_FIELD * (d_prev - d_now)
            if df != 0.0:
                reward += self._track("DIR_FIELD", df)
        self._field_prev_d, self._field_prev_goal = d_now, goal
        return reward


# --- V0.5.8 within-map refinement: empirical per-edge EXIT tiles ---------------
# The (x,y) on `from_map` at which the 100M warm-start policy crosses to
# `to_map`, derived from 60k steps of that policy (scripts/extract_exit_tiles.py;
# Route1<->Viridian modal 145-148/148, rock-solid). Only the long corridor maps
# need this — short maps (Pallet/Mart/Oak/gates) stay coarse. Edges the warm-start
# never traversed (post-parcel Pallet->Oak, Viridian->Mart) are absent -> those
# maps fall back to the coarse map-level field, refined later once reached.
_EXIT_TILE: "dict[tuple[int, int], tuple[int, int]]" = {
    (rm.MAP_PALLET_TOWN, rm.MAP_ROUTE_1):     (10, 0),    # north out of Pallet
    (rm.MAP_ROUTE_1, rm.MAP_VIRIDIAN_CITY):   (10, 0),    # north out of Route 1  ← the current stall
    (rm.MAP_VIRIDIAN_CITY, rm.MAP_ROUTE_1):   (21, 35),   # south out of Viridian (backtrack)
    (rm.MAP_ROUTE_1, rm.MAP_PALLET_TOWN):     (10, 35),   # south out of Route 1  (backtrack)
    (rm.MAP_OAKS_LAB, rm.MAP_PALLET_TOWN):    (4, 11),    # out of Oak's Lab
}


class RewardV0_5_8_fieldcount(RewardV0_5_7_fieldcount):
    """V0.5.7 + WITHIN-MAP field refinement (A's design §1.4).

    The V0.5.7 probe (jobs 52469) confirmed the coarse field DIRECTS (DIR_FIELD
    positive, wandering collapsed 49k->~100 tiles) but STALLS at ROUTE_1: the
    map-level potential is FLAT inside a map, so on the long Route 1 traverse there
    is no gradient to the Viridian exit, and the subordinate count-novelty
    (int_coef 0.25) is too weak to find it — the agent freezes rather than risk the
    -W reversal tax on a flat, tax-exposed traverse.

    Fix: add a within-map local-coordinate term so movement TOWARD the exit tile
    (one hop closer to the objective) is rewarded step-by-step:

        phi_fine(s) = -W * ( d_map(m,g) + LAMBDA * localfrac(m, x, y, g) )
        localfrac   = manhattan((x,y), EXIT_TILE[m -> next_map_toward_g]) / SCALE   in [0,1)

    LAMBDA<1 keeps a boundary crossing (delta d_map = 1) strictly dominant over any
    within-map wiggle, so the two layers never fight. Still a pure function of
    state -> telescopes identically (non-farmable: closed walk nets 0). Maps with
    no exit tile fall back to LAMBDA*0 = the coarse V0.5.7 behavior.
    """

    LAMBDA_FINE = 0.5     # within-map weight; < 1 so a map-hop always dominates
    FINE_SCALE = 48.0     # manhattan normalizer (~max on Route 1); localfrac clamped <1

    def _next_map_toward(self, m: int, goal: int):
        """Neighbor of `m` with the smallest distance to `goal` (direction of
        progress), or None if `m` is the goal / has no closer neighbor."""
        dist = _FIELD_DIST[goal]
        here = dist.get(m)
        if here is None or here == 0:
            return None
        best, best_d = None, here
        for nb in _FIELD_GRAPH.get(m, ()):
            d = dist.get(nb)
            if d is not None and d < best_d:
                best, best_d = nb, d
        return best

    def _field_dist_now(self, mem):
        goal = self._field_objective(mem)
        m = rm.map_id(mem)
        if goal is None or m not in _FIELD_DIST[goal]:
            return None, goal                       # off-graph / done
        d_map = _FIELD_DIST[goal][m]
        nxt = self._next_map_toward(m, goal)
        exit_xy = _EXIT_TILE.get((m, nxt)) if nxt is not None else None
        if exit_xy is not None:
            x, y = rm.player_position(mem)
            man = abs(x - exit_xy[0]) + abs(y - exit_xy[1])
            localfrac = min(man / self.FINE_SCALE, 0.999)   # clamp so LAMBDA*frac<1
            return d_map + self.LAMBDA_FINE * localfrac, goal
        return float(d_map), goal                   # no exit tile -> coarse fallback


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
    "RewardV0_4_3_dense": RewardV0_4_3_dense,
    "RewardV0_4_3_curated": RewardV0_4_3_curated,
    "RewardV0_4_4_dense": RewardV0_4_4_dense,
    "RewardV0_4_4_curated": RewardV0_4_4_curated,
    "RewardV0_4_5_dense": RewardV0_4_5_dense,
    "RewardV0_4_5_curated": RewardV0_4_5_curated,
    "RewardV0_5_skeleton": RewardV0_5_skeleton,
    "RewardV0_5_1_storyladder": RewardV0_5_1_storyladder,
    "RewardV0_5_2_storyladder": RewardV0_5_2_storyladder,
    "RewardV0_5_3_storyladder": RewardV0_5_3_storyladder,
    "RewardV0_5_4_storyladder": RewardV0_5_4_storyladder,
    "RewardV0_5_5_storyladder": RewardV0_5_5_storyladder,
    "RewardV0_5_7_fieldcount": RewardV0_5_7_fieldcount,
    "RewardV0_5_8_fieldcount": RewardV0_5_8_fieldcount,
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
