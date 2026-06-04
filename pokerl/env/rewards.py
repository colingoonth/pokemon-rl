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
            elif self._enemy_killed_this_battle and self._battle_was_trainer:
                reward += self.TRAINER_WIN_BONUS
            elif self._enemy_killed_this_battle or self._pokemon_caught_this_battle:
                pass  # already paid via BEAT_MON_REWARD or CATCH_REWARD
            else:
                reward += self.FLEE_PENALTY

        # ----- Per-faint penalty -----
        for i in range(min(len(party_hp), len(self._last_party_hp))):
            if self._last_party_hp[i] > 0 and party_hp[i] == 0:
                reward += self.FAINT_PENALTY
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
}


def get_reward_cls(name: str) -> type:
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward class {name!r}. "
            f"Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]
