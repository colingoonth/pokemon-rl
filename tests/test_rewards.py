"""Unit tests for RewardV0_1 using a dict-backed fake memory.

These tests don't need PyBoy or the ROM — they verify reward bookkeeping
in isolation. RAM-map address correctness is covered by test_ram_map.
"""
from __future__ import annotations

from pokerl.env import ram_map as rm
from pokerl.env.rewards import RewardV0_1


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
    r = RewardV0_1()
    r.reset(mem)
    # No state changed -> only the step penalty fires
    assert r.compute(mem) == r.STEP_PENALTY


def test_new_tile_pays_exploration_reward():
    state = base_state(x=8, y=13)
    mem = FakeMem(state)
    r = RewardV0_1()
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
    r = RewardV0_1()
    r.reset(mem)

    # Gain 2 levels in one step (e.g., big EXP gain)
    state[rm.ADDR_PARTY_MON_BASE + rm.PMON_OFFSET_LEVEL] = 8
    got = r.compute(mem)
    assert got == r.STEP_PENALTY + 2 * r.LEVEL_REWARD


def test_badge_pays_per_badge():
    state = base_state(badges=0)
    mem = FakeMem(state)
    r = RewardV0_1()
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
    r = RewardV0_1()
    r.reset(mem)
    assert r.compute(mem) == r.STEP_PENALTY


def test_unique_tiles_visited_tracks_set_size():
    state = base_state(x=8, y=13)
    mem = FakeMem(state)
    r = RewardV0_1()
    r.reset(mem)
    assert r.unique_tiles_visited == 1  # seeded with start tile

    state[rm.ADDR_PLAYER_X] = 9
    r.compute(mem)
    state[rm.ADDR_PLAYER_Y] = 14
    r.compute(mem)
    assert r.unique_tiles_visited == 3


# ---------- RewardV0_2 ----------

from pokerl.env.rewards import RewardV0_2


def _set_party_hp(state: dict[int, int], slot: int, cur: int, mx: int) -> None:
    base = rm.ADDR_PARTY_MON_BASE + slot * rm.PARTY_MON_STRUCT_SIZE
    state[base + rm.PMON_OFFSET_HP_CURRENT] = (cur >> 8) & 0xFF
    state[base + rm.PMON_OFFSET_HP_CURRENT + 1] = cur & 0xFF
    state[base + rm.PMON_OFFSET_HP_MAX] = (mx >> 8) & 0xFF
    state[base + rm.PMON_OFFSET_HP_MAX + 1] = mx & 0xFF


def test_v2_move_bonus_for_revisiting():
    state = base_state(x=8, y=13)
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)

    # Move to new tile -> +1
    state[rm.ADDR_PLAYER_X] = 9
    assert abs(r.compute(mem) - (r.STEP_PENALTY + r.EXPLORE_REWARD)) < 1e-9

    # Move back -> +0.2 (already visited)
    state[rm.ADDR_PLAYER_X] = 8
    assert abs(r.compute(mem) - (r.STEP_PENALTY + r.MOVE_BONUS)) < 1e-9


def test_v2_no_move_bonus_in_battle():
    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 20, 22)
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)

    # Enter battle; position locked
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16  # Pidgey
    state[rm.ADDR_ENEMY_MON_HP] = 0
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    # First step in battle: should fire NEW_ENCOUNTER but no move bonus
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.NEW_ENCOUNTER)) < 1e-9


def test_v2_beat_mon_reward():
    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 20, 22)
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP] = 0
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)

    # First in-battle compute pays NEW_ENCOUNTER and seeds enemy HP=30
    r.compute(mem)
    # Now enemy HP drops to 0 -> +BEAT_MON_REWARD
    state[rm.ADDR_ENEMY_MON_HP] = 0
    state[rm.ADDR_ENEMY_MON_HP + 1] = 0
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.BEAT_MON_REWARD)) < 1e-9


def test_v2_win_battle():
    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 20, 22)
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP] = 0
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)
    r.compute(mem)  # enter battle

    # Battle ends with player alive
    state[rm.ADDR_IN_BATTLE] = 0
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.WIN_BATTLE)) < 1e-9


def test_v2_lose_battle():
    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 20, 22)
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP] = 0
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)
    r.compute(mem)  # enter

    # Party member faints (-2) and battle ends with whole team down (-7)
    _set_party_hp(state, 0, 0, 22)
    state[rm.ADDR_IN_BATTLE] = 0
    got = r.compute(mem)
    # Expect step penalty + FAINT_PENALTY + LOSE_BATTLE
    expected = r.STEP_PENALTY + r.FAINT_PENALTY + r.LOSE_BATTLE
    assert abs(got - expected) < 1e-9


def test_v2_new_encounter_no_repeat():
    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 20, 22)
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)

    # First Pidgey encounter
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    got1 = r.compute(mem)
    assert abs(got1 - (r.STEP_PENALTY + r.NEW_ENCOUNTER)) < 1e-9

    # End battle, return to overworld
    state[rm.ADDR_IN_BATTLE] = 0
    r.compute(mem)

    # Second Pidgey encounter — same species, no new reward
    state[rm.ADDR_IN_BATTLE] = 1
    got2 = r.compute(mem)
    assert abs(got2 - r.STEP_PENALTY) < 1e-9


def test_v2_level_up_pays_three():
    state = base_state(level=6)
    _set_party_hp(state, 0, 20, 22)
    mem = FakeMem(state)
    r = RewardV0_2()
    r.reset(mem)

    state[rm.ADDR_PARTY_MON_BASE + rm.PMON_OFFSET_LEVEL] = 8
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + 2 * r.LEVEL_REWARD)) < 1e-9


# ---------- RewardV0_2_3 ----------

from pokerl.env.rewards import RewardV0_2_3


def test_v23_pc_heal_low_hp_pays_plus_2():
    state = base_state(map_id=rm.MAP_VIRIDIAN_POKECENTER)
    _set_party_hp(state, 0, 5, 22)  # 22% HP
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    # Baseline step at low HP — no heal event yet
    r.compute(mem)
    # Heal to full
    _set_party_hp(state, 0, 22, 22)
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.PC_HEAL_LOW)) < 1e-9


def test_v23_pc_heal_near_full_pays_minus_1():
    state = base_state(map_id=rm.MAP_VIRIDIAN_POKECENTER)
    _set_party_hp(state, 0, 20, 22)  # ~91% HP — above 80% AND above 50%
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    r.compute(mem)
    _set_party_hp(state, 0, 22, 22)
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.PC_HEAL_FULL)) < 1e-9


def test_v23_any_mon_below_50_triggers_low_branch():
    """Party total may be >80% but a single mon <50% still fires LOW."""
    state = base_state(party_count=2, map_id=rm.MAP_VIRIDIAN_POKECENTER)
    # Slot 0: 300/300, Slot 1: 49/100 -> party total 349/400 = 87% (>80%)
    # but slot 1 is below 50%, so LOW branch must fire.
    _set_party_hp(state, 0, 300, 300)
    _set_party_hp(state, 1, 49, 100)
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    r.compute(mem)  # baseline
    # Heal: both to full
    _set_party_hp(state, 1, 100, 100)
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.PC_HEAL_LOW)) < 1e-9


def test_v23_first_visit_pays_plus_5_once():
    # Start outside PC at full HP
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)

    # Walk into Viridian PC — first time
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_POKECENTER
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD     # new (map, x, y) tile
        + r.NEW_MAP_REWARD     # new map_id (parent V0.2.2)
        + r.PC_FIRST_VISIT     # new PC (V0.2.3)
    )
    assert abs(got - expected) < 1e-9

    # Leave, come back — no PC_FIRST_VISIT, no NEW_MAP, no EXPLORE
    state[rm.ADDR_MAP_ID] = rm.MAP_PALLET_TOWN
    r.compute(mem)
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_POKECENTER
    got2 = r.compute(mem)
    # Position (Viridian PC, 8, 13) already in visited -> MOVE_BONUS
    assert abs(got2 - (r.STEP_PENALTY + r.MOVE_BONUS)) < 1e-9


def test_v23_first_visit_stacks_with_low_hp_heal():
    """Single-step combo: enter new PC at low HP and heal to full pays both."""
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 5, 22)
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)

    # One step: map changes to PC AND HP heals to max
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_POKECENTER
    _set_party_hp(state, 0, 22, 22)
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD
        + r.NEW_MAP_REWARD
        + r.PC_FIRST_VISIT
        + r.PC_HEAL_LOW
    )
    assert abs(got - expected) < 1e-9


def test_v23_faint_penalty_is_minus_250():
    assert RewardV0_2_3.FAINT_PENALTY == -250.0

    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 5, 22)
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    r.compute(mem)  # enter battle (NEW_ENCOUNTER)

    # Faint + lose
    _set_party_hp(state, 0, 0, 22)
    state[rm.ADDR_IN_BATTLE] = 0
    got = r.compute(mem)
    expected = r.STEP_PENALTY + r.FAINT_PENALTY + r.LOSE_BATTLE
    assert abs(got - expected) < 1e-9


def test_v23_blackout_suppresses_heal_and_visit():
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 5, 22)
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    r.compute(mem)  # enter battle

    # Faint -> battle ends with all party dead -> blackout pending
    _set_party_hp(state, 0, 0, 22)
    state[rm.ADDR_IN_BATTLE] = 0
    r.compute(mem)
    assert r._blackout_pending is True

    # Whiteout: warp to Viridian PC with party force-healed
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_POKECENTER
    _set_party_hp(state, 0, 22, 22)
    got = r.compute(mem)

    # Heal and first-visit are suppressed. Parent rewards still fire:
    # STEP_PENALTY + EXPLORE_REWARD (new tile) + NEW_MAP_REWARD (new map_id)
    expected = r.STEP_PENALTY + r.EXPLORE_REWARD + r.NEW_MAP_REWARD
    assert abs(got - expected) < 1e-9
    assert r._blackout_pending is False
    # PC was recorded as visited so future *voluntary* visits don't pay +5
    assert rm.MAP_VIRIDIAN_POKECENTER in r._visited_pokecenters


def test_v23_reset_at_pc_preseeds_visited():
    """Spawning on a PC map_id should not later pay PC_FIRST_VISIT for it."""
    state = base_state(map_id=rm.MAP_VIRIDIAN_POKECENTER)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    assert rm.MAP_VIRIDIAN_POKECENTER in r._visited_pokecenters

    # Move tile within the PC — should only pay EXPLORE for the new tile
    state[rm.ADDR_PLAYER_X] = 9
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.EXPLORE_REWARD)) < 1e-9


def test_v23_heal_on_non_pc_map_does_not_pay():
    """Heal events outside PCs (e.g. potions in overworld) pay nothing."""
    state = base_state(map_id=rm.MAP_ROUTE_1)
    _set_party_hp(state, 0, 5, 22)
    mem = FakeMem(state)
    r = RewardV0_2_3()
    r.reset(mem)
    r.compute(mem)  # baseline
    _set_party_hp(state, 0, 22, 22)
    got = r.compute(mem)
    # No PC_HEAL_* fires off-map. Only step penalty.
    assert abs(got - r.STEP_PENALTY) < 1e-9


# ---------- RewardV0_2_4 family (faint-magnitude sweep) ----------

from pokerl.env.rewards import (
    RewardV0_2_4_f25,
    RewardV0_2_4_f50,
    RewardV0_2_4_f100,
)


def test_v24_f25_faint_magnitude():
    assert RewardV0_2_4_f25.FAINT_PENALTY == -25.0
    assert RewardV0_2_4_f25.PC_FIRST_VISIT == 5.0  # inherits from V0.2.3
    assert RewardV0_2_4_f25.PC_HEAL_LOW == 2.0


def test_v24_f50_faint_magnitude():
    assert RewardV0_2_4_f50.FAINT_PENALTY == -50.0


def test_v24_f100_faint_magnitude():
    assert RewardV0_2_4_f100.FAINT_PENALTY == -100.0


def test_v24_faint_penalty_actually_fires_at_overridden_magnitude():
    """End-to-end: lose-battle scenario produces the per-variant magnitude."""
    for cls, expected_penalty in [
        (RewardV0_2_4_f25, -25.0),
        (RewardV0_2_4_f50, -50.0),
        (RewardV0_2_4_f100, -100.0),
    ]:
        state = base_state(x=8, y=13)
        _set_party_hp(state, 0, 5, 22)
        state[rm.ADDR_IN_BATTLE] = 1
        state[rm.ADDR_ENEMY_MON_SPECIES] = 16
        state[rm.ADDR_ENEMY_MON_HP + 1] = 30
        mem = FakeMem(state)
        r = cls()
        r.reset(mem)
        r.compute(mem)  # enter battle
        _set_party_hp(state, 0, 0, 22)
        state[rm.ADDR_IN_BATTLE] = 0
        got = r.compute(mem)
        expected = r.STEP_PENALTY + expected_penalty + r.LOSE_BATTLE
        assert abs(got - expected) < 1e-9, (
            f"{cls.__name__}: got {got}, expected {expected}"
        )


# ---------- RewardV0_2_5 family (building / mart bonuses) ----------

from pokerl.env.rewards import (
    RewardV0_2_5_b20,
    RewardV0_2_5_b50,
    RewardV0_2_5_b100,
)


def test_v25_magnitudes():
    assert RewardV0_2_5_b20.NEW_MAP_REWARD == 20.0
    assert RewardV0_2_5_b50.NEW_MAP_REWARD == 50.0
    assert RewardV0_2_5_b100.NEW_MAP_REWARD == 100.0
    # Inherits other constants from V0.2.5 -> V0.2.4_f25 -> V0.2.3
    assert RewardV0_2_5_b20.FAINT_PENALTY == -25.0
    assert RewardV0_2_5_b20.MART_PC_BONUS == 20.0
    assert RewardV0_2_5_b20.PC_HEAL_LOW == 5.0


def test_v25_pokemart_first_visit_stacks():
    """First-time entry into a Pokemart pays NEW_MAP + MART_PC_BONUS."""
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_5_b20()
    r.reset(mem)

    # Walk into Viridian Mart
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_MART
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD          # new (map, x, y) tile
        + r.NEW_MAP_REWARD          # +20
        + r.MART_PC_BONUS           # +20
    )
    assert abs(got - expected) < 1e-9


def test_v25_pokecenter_first_visit_triple_stacks():
    """First PC entry stacks NEW_MAP + MART_PC_BONUS + PC_FIRST_VISIT."""
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_5_b50()
    r.reset(mem)

    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_POKECENTER
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD
        + r.NEW_MAP_REWARD          # +50
        + r.MART_PC_BONUS           # +20
        + r.PC_FIRST_VISIT          # +5
    )
    assert abs(got - expected) < 1e-9


def test_v25_non_special_building_only_new_map():
    """Entering a building that's NOT a Mart/PC pays only NEW_MAP_REWARD."""
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_5_b100()
    r.reset(mem)

    # Oak's Lab is a building but not a Mart or PC
    state[rm.ADDR_MAP_ID] = rm.MAP_OAKS_LAB
    got = r.compute(mem)
    expected = r.STEP_PENALTY + r.EXPLORE_REWARD + r.NEW_MAP_REWARD  # +100
    assert abs(got - expected) < 1e-9


def test_v25_pc_heal_low_bumped_to_5():
    """V0.2.5 raises PC_HEAL_LOW to 5.0 (was 2.0 in V0.2.3)."""
    state = base_state(map_id=rm.MAP_VIRIDIAN_POKECENTER)
    _set_party_hp(state, 0, 5, 22)  # 22% HP, low
    mem = FakeMem(state)
    r = RewardV0_2_5_b20()
    r.reset(mem)
    r.compute(mem)  # baseline at low HP
    _set_party_hp(state, 0, 22, 22)  # heal to full
    got = r.compute(mem)
    # PC was pre-seeded into _visited_pokecenters at reset, so no PC_FIRST_VISIT
    # No new map_id since we started here.
    assert abs(got - (r.STEP_PENALTY + r.PC_HEAL_LOW)) < 1e-9
    assert r.PC_HEAL_LOW == 5.0


# ---------- RewardV0_2_6 (small Mart/PC + Gym bonuses, restore f25 balance) ----------

from pokerl.env.rewards import RewardV0_2_6


def test_v26_inherits_v24_f25_magnitudes():
    """V0.2.6 must not regress f25's combat balance."""
    r = RewardV0_2_6
    assert r.FAINT_PENALTY == -25.0          # from V0.2.4_f25
    assert r.NEW_MAP_REWARD == 5.0           # from V0.2.2 (unchanged)
    assert r.BEAT_MON_REWARD == 10.0         # from V0.2.2 (unchanged)
    assert r.PC_HEAL_LOW == 2.0              # back to V0.2.3 default (V0.2.5 bumped to 5; V0.2.6 restores)
    assert r.MART_PC_BONUS == 10.0
    assert r.GYM_BONUS == 20.0


def test_v26_mart_first_visit_pays_stacked_bonus():
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_6()
    r.reset(mem)
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_MART
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD   # +0.3, new (map, x, y)
        + r.NEW_MAP_REWARD   # +5
        + r.MART_PC_BONUS    # +10
    )
    assert abs(got - expected) < 1e-9


def test_v26_pewter_gym_first_visit_pays_gym_bonus():
    """Brock's gym is the V1 target — first visit pays NEW_MAP + GYM."""
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_6()
    r.reset(mem)
    state[rm.ADDR_MAP_ID] = rm.MAP_PEWTER_GYM
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD
        + r.NEW_MAP_REWARD   # +5
        + r.GYM_BONUS        # +20
    )
    assert abs(got - expected) < 1e-9


def test_v26_pc_first_visit_pays_pc_plus_mart_pc_bonus():
    """PC first-visit: NEW_MAP + MART_PC_BONUS + PC_FIRST_VISIT."""
    state = base_state(map_id=rm.MAP_PALLET_TOWN, x=8, y=13)
    _set_party_hp(state, 0, 22, 22)
    mem = FakeMem(state)
    r = RewardV0_2_6()
    r.reset(mem)
    state[rm.ADDR_MAP_ID] = rm.MAP_VIRIDIAN_POKECENTER
    got = r.compute(mem)
    expected = (
        r.STEP_PENALTY
        + r.EXPLORE_REWARD
        + r.NEW_MAP_REWARD   # +5
        + r.MART_PC_BONUS    # +10
        + r.PC_FIRST_VISIT   # +5 (inherited V0.2.3)
    )
    assert abs(got - expected) < 1e-9


def test_v26_gym_and_mart_lists_disjoint():
    """Gym map_ids must not overlap with Mart/PC ids — otherwise a single
    map would pay both MART_PC_BONUS and GYM_BONUS, which is a config bug."""
    assert rm.GYM_MAP_IDS.isdisjoint(rm.POKEMART_MAP_IDS)
    assert rm.GYM_MAP_IDS.isdisjoint(rm.POKECENTER_MAP_IDS)
    # And the V1 target is actually in the gym set.
    assert rm.MAP_PEWTER_GYM in rm.GYM_MAP_IDS


# ---------- RewardV0_2_7 (combat bump) ----------

from pokerl.env.rewards import RewardV0_2_7


def test_v27_combat_magnitudes_bumped():
    assert RewardV0_2_7.BEAT_MON_REWARD == 20.0
    assert RewardV0_2_7.TRAINER_WIN_BONUS == 30.0
    # All other f25 / V0.2.6 inheritance preserved
    assert RewardV0_2_7.FAINT_PENALTY == -25.0
    assert RewardV0_2_7.NEW_MAP_REWARD == 5.0
    assert RewardV0_2_7.MART_PC_BONUS == 10.0
    assert RewardV0_2_7.GYM_BONUS == 20.0


def test_v27_wild_kill_pays_20():
    """Killing a wild Pokemon now pays +20 (was +10)."""
    state = base_state(x=8, y=13)
    _set_party_hp(state, 0, 20, 22)
    state[rm.ADDR_IN_BATTLE] = 1
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16  # Pidgey
    state[rm.ADDR_ENEMY_MON_HP + 1] = 30
    mem = FakeMem(state)
    r = RewardV0_2_7()
    r.reset(mem)
    r.compute(mem)  # enter battle (NEW_ENCOUNTER)
    # Enemy HP drops to 0
    state[rm.ADDR_ENEMY_MON_HP] = 0
    state[rm.ADDR_ENEMY_MON_HP + 1] = 0
    got = r.compute(mem)
    assert abs(got - (r.STEP_PENALTY + r.BEAT_MON_REWARD)) < 1e-9
    assert r.BEAT_MON_REWARD == 20.0


# --- V0.4.4 PC_FIRST_VISIT heal-entry mixin ----------------------------------

def test_v044_pc_first_visit_fires_once_on_center_entry():
    """Entering a Pokecenter map pays PC_FIRST_VISIT exactly once; re-entry
    (and staying) pays nothing. Verified on both A/B arms."""
    from pokerl.env.rewards import RewardV0_4_4_dense, RewardV0_4_4_curated
    pc = rm.MAP_VIRIDIAN_POKECENTER
    for cls in (RewardV0_4_4_dense, RewardV0_4_4_curated):
        r = cls()
        r.reset(FakeMem(base_state(map_id=rm.MAP_ROUTE_1)))  # start outside a PC
        # step in the overworld: no PC reward
        r.compute(FakeMem(base_state(map_id=rm.MAP_ROUTE_1, x=9)))
        assert "PC_FIRST_VISIT" not in r.last_components, cls.__name__
        # enter the Center: fires once, +10
        r.compute(FakeMem(base_state(map_id=pc)))
        assert r.last_components.get("PC_FIRST_VISIT") == 10.0, cls.__name__
        # still inside / re-enter: never pays again
        r.compute(FakeMem(base_state(map_id=pc, x=9)))
        assert "PC_FIRST_VISIT" not in r.last_components, cls.__name__


def test_v044_pc_first_visit_premasked_if_started_in_center():
    """Defensive: starting inside a PC pre-masks it (no free reward)."""
    from pokerl.env.rewards import RewardV0_4_4_curated
    pc = rm.MAP_VIRIDIAN_POKECENTER
    r = RewardV0_4_4_curated()
    r.reset(FakeMem(base_state(map_id=pc)))
    r.compute(FakeMem(base_state(map_id=pc, x=9)))
    assert "PC_FIRST_VISIT" not in r.last_components


# --- V0.5 hard-gate readers + thin skeleton ---

def _mem_with_byte(addr: int, value: int) -> "FakeMem":
    s = base_state()
    s[addr] = value
    return FakeMem(s)


def test_hard_gate_readers_bit_positions():
    # parcel = 0xD74E bit1, pokedex = 0xD74B bit5, brock = 0xD755 bit7
    assert rm.got_oaks_parcel(_mem_with_byte(0xD74E, 1 << 1)) is True
    assert rm.got_oaks_parcel(FakeMem(base_state())) is False
    assert rm.got_pokedex(_mem_with_byte(0xD74B, 1 << 5)) is True
    # bit 4 (GOT_POKEBALLS_FROM_OAK) must NOT read as pokedex
    assert rm.got_pokedex(_mem_with_byte(0xD74B, 1 << 4)) is False
    assert rm.beat_brock(_mem_with_byte(0xD755, 1 << 7)) is True
    assert rm.beat_brock(FakeMem(base_state())) is False


def test_v05_skeleton_disables_exploration_shaping():
    from pokerl.env.rewards import RewardV0_5_skeleton
    assert RewardV0_5_skeleton.EXPLORE_COEF == 0.0
    assert RewardV0_5_skeleton.STUCK_PENALTY_PER_STEP == 0.0


def test_v05_gate_fires_once_then_masks():
    from pokerl.env.rewards import RewardV0_5_skeleton
    r = RewardV0_5_skeleton()
    r.reset(FakeMem(base_state()))           # no gates satisfied at start
    mem = _mem_with_byte(0xD74E, 1 << 1)     # parcel obtained
    r.compute(mem)
    assert r.last_components.get("GATE_GOT_OAKS_PARCEL") == 10.0
    # second step with the flag still set -> no double-pay
    r.compute(mem)
    assert "GATE_GOT_OAKS_PARCEL" not in r.last_components


def test_v05_gate_premasked_if_satisfied_at_start():
    from pokerl.env.rewards import RewardV0_5_skeleton
    r = RewardV0_5_skeleton()
    r.reset(_mem_with_byte(0xD74E, 1 << 1))  # already have the parcel at reset
    r.compute(_mem_with_byte(0xD74E, 1 << 1))
    assert "GATE_GOT_OAKS_PARCEL" not in r.last_components


def test_v05_beat_brock_gate_pays_fifty():
    from pokerl.env.rewards import RewardV0_5_skeleton
    r = RewardV0_5_skeleton()
    r.reset(FakeMem(base_state()))
    r.compute(_mem_with_byte(0xD755, 1 << 7))
    assert r.last_components.get("GATE_BEAT_BROCK") == 50.0


# --- V0.5.1 story-multiplier ladder ---

def test_storyladder_registered_and_constructs():
    from pokerl.env.rewards import get_reward_cls, RewardV0_5_1_storyladder
    assert get_reward_cls("RewardV0_5_1_storyladder") is RewardV0_5_1_storyladder


def test_storyladder_starts_at_unit_multiplier():
    from pokerl.env.rewards import RewardV0_5_1_storyladder
    r = RewardV0_5_1_storyladder()
    r.reset(FakeMem(base_state(map_id=0x00)))   # Pallet, nothing achieved
    assert r._mult == 1.0
    assert r._fired == set()


def test_storyladder_map_rung_fires_once_and_raises_mult():
    from pokerl.env.rewards import RewardV0_5_1_storyladder
    r = RewardV0_5_1_storyladder()
    r.reset(FakeMem(base_state(map_id=0x00)))
    mem = FakeMem(base_state(map_id=0x0C))      # entered Route 1
    r.compute(mem)
    assert r.last_components.get("RUNG_ROUTE_1") == 4.0
    assert abs(r._mult - 1.10) < 1e-9
    # revisit the same map -> no double-pay, multiplier unchanged
    r.compute(mem)
    assert "RUNG_ROUTE_1" not in r.last_components
    assert abs(r._mult - 1.10) < 1e-9


def test_storyladder_pokedex_flag_rung_is_the_keystone():
    from pokerl.env.rewards import RewardV0_5_1_storyladder
    r = RewardV0_5_1_storyladder()
    r.reset(FakeMem(base_state(map_id=0x00)))
    r.compute(_mem_with_byte(0xD74B, 1 << 5))   # got Pokedex (the north-gate key)
    assert r.last_components.get("RUNG_POKEDEX") == 30.0
    assert abs(r._mult - 1.50) < 1e-9


def test_storyladder_premasks_progress_at_reset():
    from pokerl.env.rewards import RewardV0_5_1_storyladder
    r = RewardV0_5_1_storyladder()
    # warm-start already holding the parcel, on a neutral (non-ladder) map so
    # only the PARCEL rung pre-masks: multiplier seeded, no one-shot payout.
    s = base_state(map_id=0x70)         # 0x70 is not a ladder map
    s[0xD74E] = 1 << 1                   # parcel obtained
    r.reset(FakeMem(s))
    assert "PARCEL" in r._fired
    assert abs(r._mult - 1.20) < 1e-9
    r.compute(FakeMem(s))
    assert "RUNG_PARCEL" not in r.last_components


def test_storyladder_backtrack_staircase():
    """The parcel->Oak return leg lights up progress-conditioned rungs that
    plain first-entry map rungs miss (the maps were already visited outbound)."""
    from pokerl.env.rewards import RewardV0_5_1_storyladder
    r = RewardV0_5_1_storyladder()
    r.reset(FakeMem(base_state(map_id=0x70)))   # neutral start, nothing fired

    def mem(map_id):
        s = base_state(map_id=map_id)
        s[0xD74E] = 1 << 1                       # carrying the parcel
        return FakeMem(s)

    r.compute(mem(0x0C))                          # back on Route 1 with the parcel
    assert r.last_components.get("RUNG_RETURN_ROUTE_1") == 6.0
    r.compute(mem(0x00))                          # into Pallet with the parcel
    assert r.last_components.get("RUNG_RETURN_PALLET") == 8.0
    r.compute(mem(0x28))                          # at Oak's Lab, the delivery point
    assert r.last_components.get("RUNG_AT_OAK_PARCEL") == 12.0
    assert {"RETURN_ROUTE_1", "RETURN_PALLET", "AT_OAK_PARCEL"} <= r._fired


# --- V0.5.2 potential-based danger-zone HP (replaces farmable quadratic heal) ---

def _hp_state(cur: int, mx: int, **kw) -> "FakeMem":
    """base_state with one party mon at cur/mx HP (u16 big-endian fields)."""
    s = base_state(party_count=1, **kw)
    base = rm.ADDR_PARTY_MON_BASE
    s[base + rm.PMON_OFFSET_HP_CURRENT] = (cur >> 8) & 0xFF
    s[base + rm.PMON_OFFSET_HP_CURRENT + 1] = cur & 0xFF
    s[base + rm.PMON_OFFSET_HP_MAX] = (mx >> 8) & 0xFF
    s[base + rm.PMON_OFFSET_HP_MAX + 1] = mx & 0xFF
    return FakeMem(s)


def test_v52_quadratic_heal_damage_disabled():
    from pokerl.env.rewards import RewardV0_5_2_storyladder
    assert RewardV0_5_2_storyladder.HEAL_QUAD_COEF == 0.0
    assert RewardV0_5_2_storyladder.DAMAGE_QUAD_COEF == 0.0


def test_v52_no_hp_penalty_when_healthy():
    """Above the 50% danger threshold the HP term is flat -> a won fight that
    ends healthy costs nothing (the anti-timidity property)."""
    from pokerl.env.rewards import RewardV0_5_2_storyladder
    r = RewardV0_5_2_storyladder()
    r.reset(_hp_state(100, 100))          # full HP
    r.compute(_hp_state(70, 100))         # took damage but still > 50%
    assert "HP_DANGER" not in r.last_components


def test_v52_danger_zone_debt_and_potential_round_trip():
    """Dropping into the danger zone costs phi; healing back repays it exactly
    (potential telescopes -> hurt->heal nets ~0, so no farm)."""
    from pokerl.env.rewards import RewardV0_5_2_storyladder
    r = RewardV0_5_2_storyladder()
    r.reset(_hp_state(100, 100))
    r.compute(_hp_state(20, 100))         # 1.0 -> 0.2: phi(0.2)-phi(1.0) = -8*0.3^2
    debt = r.last_components["HP_DANGER"]
    assert abs(debt - (-0.72)) < 1e-6
    r.compute(_hp_state(100, 100))        # 0.2 -> 1.0: repays +0.72
    repay = r.last_components["HP_DANGER"]
    assert abs(repay - 0.72) < 1e-6
    assert abs(debt + repay) < 1e-9       # round trip nets zero -> unfarmable


def test_v52_respawn_jump_is_skipped():
    """The faint->respawn HP jump (prev == 0 -> full) must not pay a heal
    repayment, or fainting could be partially refunded."""
    from pokerl.env.rewards import RewardV0_5_2_storyladder
    r = RewardV0_5_2_storyladder()
    r.reset(_hp_state(100, 100))
    r.compute(_hp_state(0, 100))          # fainted: prev 1.0 -> now 0.0
    r.compute(_hp_state(100, 100))        # respawn: prev 0.0 -> now 1.0, must skip
    assert "HP_DANGER" not in r.last_components


# --- V0.5.3: rival-win bonus + escalating battle-stall ---

def _battle_mem(party_cur=50, party_mx=50, enemy_hp=20, **kw) -> "FakeMem":
    """In-battle (trainer) FakeMem with fixed party + enemy HP for stall tests."""
    s = base_state(party_count=1, **kw)
    base = rm.ADDR_PARTY_MON_BASE
    s[base + rm.PMON_OFFSET_HP_CURRENT] = (party_cur >> 8) & 0xFF
    s[base + rm.PMON_OFFSET_HP_CURRENT + 1] = party_cur & 0xFF
    s[base + rm.PMON_OFFSET_HP_MAX] = (party_mx >> 8) & 0xFF
    s[base + rm.PMON_OFFSET_HP_MAX + 1] = party_mx & 0xFF
    s[rm.ADDR_IN_BATTLE] = 2               # trainer battle
    s[rm.ADDR_ENEMY_MON_HP] = (enemy_hp >> 8) & 0xFF
    s[rm.ADDR_ENEMY_MON_HP + 1] = enemy_hp & 0xFF
    return FakeMem(s)


def test_v53_rival_win_rung_is_first_and_fires():
    from pokerl.env.rewards import RewardV0_5_3_storyladder
    assert RewardV0_5_3_storyladder._LADDER[0][0] == "RIVAL_WIN"
    r = RewardV0_5_3_storyladder()
    r.reset(FakeMem(base_state(map_id=0x28)))   # Oak's lab, rival flag unset
    r.compute(_mem_with_byte(0xD74B, 1 << 3))   # rival battle won -> flag bit 3
    assert r.last_components.get("RUNG_RIVAL_WIN") == 25.0


def test_v53_battle_stall_escalates_then_caps():
    """Sustained stalling grows the per-step penalty (-0.01 -> -1.0) instead of
    the flat -0.01, then caps at COEF*RAMP."""
    from pokerl.env.rewards import RewardV0_5_3_storyladder
    r = RewardV0_5_3_storyladder()
    mem = _battle_mem()
    r.reset(mem)
    pens = []
    for _ in range(200):                  # same mem each step -> no HP change -> stall climbs
        r.compute(mem)
        pens.append(r.last_components.get("BATTLE_STALL", 0.0))
    nz = [p for p in pens if p != 0.0]
    assert nz, "stall penalty never fired"
    assert abs(nz[-1]) > abs(nz[0])                       # escalates
    assert any(abs(p - (-1.0)) < 1e-9 for p in nz)        # reaches the cap -0.01*100
    assert min(nz) >= -1.0 - 1e-9                          # never exceeds the cap


def test_v53_base_class_keeps_flat_stall():
    """RAMP defaults to 1 (flat) so existing classes are unchanged."""
    from pokerl.env.rewards import RewardV0_4_2_center, RewardV0_5_3_storyladder
    assert RewardV0_4_2_center.BATTLE_STALL_RAMP == 1
    assert RewardV0_5_3_storyladder.BATTLE_STALL_RAMP == 100


# --- V0.5.4: dense enemy-damage engagement reward ---

def test_v54_enemy_damage_rewarded():
    from pokerl.env.rewards import RewardV0_5_4_storyladder
    r = RewardV0_5_4_storyladder()
    r.reset(_battle_mem(enemy_hp=20))      # in trainer battle, enemy at 20 HP
    r.compute(_battle_mem(enemy_hp=12))    # enemy took 8 damage
    assert abs(r.last_components.get("ENEMY_DMG", 0.0) - 0.15 * 8) < 1e-9


def test_v54_no_reward_for_enemy_hp_increase():
    """A fresh enemy sent in (HP jumps up) must not pay damage reward."""
    from pokerl.env.rewards import RewardV0_5_4_storyladder
    r = RewardV0_5_4_storyladder()
    r.reset(_battle_mem(enemy_hp=5))
    r.compute(_battle_mem(enemy_hp=18))    # HP up -> negative delta -> no reward
    assert "ENEMY_DMG" not in r.last_components


def test_v54_stall_ramp_reverted_to_flat():
    from pokerl.env.rewards import RewardV0_5_4_storyladder
    assert RewardV0_5_4_storyladder.BATTLE_STALL_RAMP == 1
    assert RewardV0_5_4_storyladder._LADDER[0][0] == "RIVAL_WIN"   # still inherits the capstone


def test_storyladder_multiplier_scales_combat_base():
    """STORY_MULT contributes (M-1)*base so the breakdown sums to M*base, i.e.
    combat is literally worth more once the story has progressed."""
    from pokerl.env.rewards import RewardV0_5_1_storyladder
    r = RewardV0_5_1_storyladder()
    r.reset(FakeMem(base_state(map_id=0x00, level=6)))
    r.compute(_mem_with_byte(0xD74B, 1 << 5))            # fire Pokedex -> M = 1.5
    # a level-up step yields a positive combat base and fires no new rung
    total = r.compute(FakeMem(base_state(map_id=0x00, level=8)))
    base = sum(v for k, v in r.last_components.items()
               if k != "STORY_MULT" and not k.startswith("RUNG_"))
    assert base != 0.0                                   # the test is meaningful
    assert abs(r.last_components.get("STORY_MULT", 0.0) - 0.5 * base) < 1e-9
    assert abs(total - 1.5 * base) < 1e-9                # no new rung this step


# ---------------------------------------------------------------------------
# V0.5.5 corridor-stabilizing fix: per-area ENEMY_DMG diminishing + coupled
# flee relief. Both driven by one shared per-area engagement budget.
# ---------------------------------------------------------------------------
from pokerl.env.rewards import RewardV0_5_5_storyladder


def test_v0_5_5_enemy_dmg_diminishes_per_area():
    """ENEMY_DMG pays full coef in a fresh area, decays linearly to 0 as the
    area's budget fills, and resets when the agent enters a NEW map."""
    r = RewardV0_5_5_storyladder()
    B = r.ENEMY_DMG_BUDGET_PER_AREA            # 100.0
    C = r.ENEMY_DMG_COEF                        # 0.15
    mem = FakeMem(base_state(map_id=0x12))     # Route 1
    r.reset(mem)

    # Fresh area (spent=0): full coefficient.
    assert abs(r._enemy_dmg_amount(40.0, mem) - C * 40.0 * 1.0) < 1e-9
    # spent=40, next hit's remaining fraction = 1 - 40/100 = 0.6
    assert abs(r._enemy_dmg_amount(40.0, mem) - C * 40.0 * 0.6) < 1e-9
    # spent=80 -> 100: budget saturated, further grinding in this area pays 0.
    r._enemy_dmg_amount(40.0, mem)             # spent now 120 (>= B)
    assert r._enemy_dmg_amount(40.0, mem) == 0.0

    # A DIFFERENT map is fresh again -> full coefficient.
    mem2 = FakeMem(base_state(map_id=0x13))    # Viridian
    assert abs(r._enemy_dmg_amount(40.0, mem2) - C * 40.0 * 1.0) < 1e-9


def _enter_wild_battle(state, enemy_hp):
    state[rm.ADDR_IN_BATTLE] = 1               # 1 = wild
    state[rm.ADDR_ENEMY_MON_SPECIES] = 16
    state[rm.ADDR_ENEMY_MON_HP] = enemy_hp >> 8
    state[rm.ADDR_ENEMY_MON_HP + 1] = enemy_hp & 0xFF


def test_v0_5_5_flee_penalized_in_fresh_area():
    """In an un-engaged area the flee penalty still fires — fighting is
    incentivized before the agent has cleared the area."""
    state = base_state(map_id=0x12)
    _set_party_hp(state, 0, 20, 22)
    mem = FakeMem(state)
    r = RewardV0_5_5_storyladder()
    r.reset(mem)

    _enter_wild_battle(state, 30)
    r.compute(mem)                             # enter battle, no damage dealt
    state[rm.ADDR_IN_BATTLE] = 0               # flee (enemy alive, party alive)
    r.compute(mem)

    assert r.last_components.get("FLEE_PENALTY", 0.0) < 0.0


def test_v0_5_5_flee_free_after_area_cleared():
    """Once the area's engagement budget is spent, fleeing its wild encounters
    is refunded (free) so the agent can traverse instead of being trapped."""
    state = base_state(map_id=0x12)
    _set_party_hp(state, 0, 20, 22)
    mem = FakeMem(state)
    r = RewardV0_5_5_storyladder()
    r.reset(mem)

    _enter_wild_battle(state, 150)
    r.compute(mem)                             # enter battle, enemy HP 150
    state[rm.ADDR_ENEMY_MON_HP + 1] = 49       # deal 101 dmg, enemy still alive
    r.compute(mem)
    assert r._enemy_dmg_per_area.get(0x12, 0.0) >= r.ENEMY_DMG_BUDGET_PER_AREA

    state[rm.ADDR_IN_BATTLE] = 0               # flee from the cleared area
    r.compute(mem)
    assert "FLEE_PENALTY" not in r.last_components      # refunded
