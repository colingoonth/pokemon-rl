"""RAM addresses + read helpers for Pokemon Red (US).

Addresses derived from the pret/pokered disassembly. WRAM lives at
0xC000-0xDFFF on the Game Boy; almost everything we care about is in
the 0xD000+ range.

These constants are mid-confidence — most are well-documented across
multiple Pokemon Red RL writeups, but a few are best-guess. The
companion verify script (pokerl.scripts.verify_save_state) loads
states/post_intro.state and prints all reads so we can sanity-check
empirically before trusting any of them in a reward function.
"""
from __future__ import annotations

from typing import Protocol


class MemoryView(Protocol):
    """Anything that supports `mem[addr]` and `mem[a:b]` byte reads.

    pyboy.PyBoy.memory satisfies this. Lets us keep this module pyboy-free
    so it's trivially unit-testable with a dict or bytes mock.
    """

    def __getitem__(self, key):  # pragma: no cover
        ...


# --- Addresses (hex, from pret/pokered) ---

ADDR_PARTY_COUNT       = 0xD163  # u8, number of Pokemon in party (0-6)
ADDR_PARTY_SPECIES     = 0xD164  # 6 bytes of species ids, then 0xFF terminator
ADDR_PARTY_MON_BASE    = 0xD16B  # start of first party-mon struct
PARTY_MON_STRUCT_SIZE  = 44      # bytes per party mon

# Offsets within a single party-mon struct (44 bytes total)
PMON_OFFSET_HP_CURRENT = 0x01    # u16 big-endian, current HP
PMON_OFFSET_LEVEL      = 0x21    # u8, current level (offset 33)
PMON_OFFSET_HP_MAX     = 0x22    # u16 big-endian, max HP (offset 34)

ADDR_BADGES            = 0xD356  # u8 bitfield: bit i set = badge i earned
ADDR_MAP_ID            = 0xD35E  # u8, current map id (0x00 = Pallet Town)
ADDR_PLAYER_Y          = 0xD361  # u8, Y coordinate on current map
ADDR_PLAYER_X          = 0xD362  # u8, X coordinate on current map

ADDR_MONEY             = 0xD347  # 3 bytes BCD (big-endian)
ADDR_IN_BATTLE         = 0xD057  # u8, 0=overworld, 1=wild, 2=trainer

# Battle-time addresses. These are read during in_battle != 0.
# All best-guess from pret/pokered; verify with verify_battle_state before
# trusting in a reward function.
ADDR_ENEMY_MON_SPECIES = 0xCFE5  # u8, species id of current enemy mon
ADDR_ENEMY_MON_HP      = 0xCFE6  # u16 big-endian, enemy mon current HP
ADDR_TRAINER_CLASS     = 0xD059  # u8, trainer class id (e.g. JR_TRAINER_F)
ADDR_TRAINER_NUMBER    = 0xD05D  # u8, specific trainer within the class

# Event flags: a long bitarray. Each story flag is one bit somewhere in here.
ADDR_EVENT_FLAGS_START = 0xD747
ADDR_EVENT_FLAGS_END   = 0xD886  # inclusive
EVENT_FLAGS_BYTES      = ADDR_EVENT_FLAGS_END - ADDR_EVENT_FLAGS_START + 1

# Hard-gate story events on the path to Brock, as (byte_addr, bit) pairs.
# Verified twice against pret/pokered constants/event_constants.asm AND
# drubinstein/pokemonred_puffer data/events.py (both agree exactly):
#   addr = 0xD747 + index//8,  bit = index%8.
# These are the "true gates" the V0.5 thin reward pays a one-shot bonus for
# (objectives, not paths). The parcel-delivery / pokedex pair is the gate
# that forces the Pallet backtrack; BEAT_BROCK is the V1 goal.
HARD_GATE_GOT_OAKS_PARCEL = (0xD74E, 1)   # EVENT_GOT_OAKS_PARCEL,  index 57
HARD_GATE_GOT_POKEDEX     = (0xD74B, 5)   # EVENT_GOT_POKEDEX,      index 37 (bit 5, NOT 4)
HARD_GATE_BEAT_BROCK      = (0xD755, 7)   # EVENT_BEAT_BROCK,       index 119
# Not a reward gate, but a useful hidden-progress bit for the RND context
# vector: marks the post-Oak's-Lab state (index 36, the bit adjacent to POKEDEX).
EVENT_GOT_POKEBALLS_FROM_OAK = (0xD74B, 4)

# Intermediate storyladder rungs (Day-11, verified against BOTH pret/pokered
# event_constants.asm AND drubinstein/pokemonred_puffer events.py):
EVENT_BATTLED_RIVAL_IN_OAKS_LAB = (0xD74B, 3)  # rival battle WON (the start-state opener), index 35
EVENT_GOT_POTION_SAMPLE = (0xD75F, 0)   # Viridian Mart clerk free sample, index 192 — early Viridian beat
EVENT_GOT_TM34          = (0xD755, 6)   # Brock's TM reward, index 118 — fires with the Brock win

# Pokedex owned bitfield (19 bytes, 1 bit per species, 151 species).
ADDR_POKEDEX_OWNED_START = 0xD2F7
ADDR_POKEDEX_OWNED_END   = 0xD309  # inclusive

# Bag inventory: count byte at 0xD31D, then [item_id, qty] pairs starting at 0xD31E,
# terminated by 0xFF after the last pair. Max 20 items.
ADDR_BAG_COUNT = 0xD31D
ADDR_BAG_ITEMS = 0xD31E

# Item IDs (from pret/pokered constants/item_constants.asm)
ITEM_POKEBALL_ID = 0x04

# Known map ids (handful — full list in pret/pokered constants/map_constants.asm)
MAP_PALLET_TOWN        = 0x00
MAP_VIRIDIAN_CITY      = 0x01
MAP_PEWTER_CITY        = 0x02
MAP_ROUTE_1            = 0x0C
MAP_ROUTE_2            = 0x0D
MAP_OAKS_LAB           = 0x28
# Brock-path maps north of Viridian. Verified Day-11 against BOTH pret/pokered
# constants/map_constants.asm AND drubinstein/pokemonred_puffer. Reachable only
# after EVENT_GOT_POKEDEX (the Viridian old-man gate); the south gate (0x32) is
# the V0.5.1 storyladder GOAL — first map past the wedge.
MAP_VIRIDIAN_FOREST_SOUTH_GATE = 0x32
MAP_VIRIDIAN_FOREST            = 0x33
MAP_VIRIDIAN_FOREST_NORTH_GATE = 0x2F

# Pokemon Center map ids (from pret/pokered constants/map_constants.asm).
# Viridian + Pewter are V1-critical (Brock route); others included for forward compat.
MAP_VIRIDIAN_POKECENTER  = 0x29
MAP_PEWTER_POKECENTER    = 0x3A
MAP_CERULEAN_POKECENTER  = 0x40
MAP_VERMILION_POKECENTER = 0x59
MAP_CELADON_POKECENTER   = 0x83
MAP_LAVENDER_POKECENTER  = 0x8D
MAP_FUCHSIA_POKECENTER   = 0x9A
MAP_CINNABAR_POKECENTER  = 0xAB
MAP_INDIGO_POKECENTER    = 0xAE
MAP_SAFFRON_POKECENTER   = 0xB6

POKECENTER_MAP_IDS: frozenset[int] = frozenset({
    MAP_VIRIDIAN_POKECENTER,
    MAP_PEWTER_POKECENTER,
    MAP_CERULEAN_POKECENTER,
    MAP_VERMILION_POKECENTER,
    MAP_CELADON_POKECENTER,
    MAP_LAVENDER_POKECENTER,
    MAP_FUCHSIA_POKECENTER,
    MAP_CINNABAR_POKECENTER,
    MAP_INDIGO_POKECENTER,
    MAP_SAFFRON_POKECENTER,
})

# Pokemart map ids (pret/pokered). Viridian + Pewter are V1-critical.
# Celadon Dept Store is a multi-floor megamart with different map_ids per
# floor — included primary lobby only; sub-floors will register as their
# own new map_ids and pay the base NEW_MAP_REWARD without the mart bonus.
MAP_VIRIDIAN_MART  = 0x2A
MAP_PEWTER_MART    = 0x3B
MAP_CERULEAN_MART  = 0x41
MAP_VERMILION_MART = 0x5A
MAP_LAVENDER_MART  = 0x8E
MAP_FUCHSIA_MART   = 0x9B
MAP_CINNABAR_MART  = 0xAC
MAP_SAFFRON_MART   = 0xB7

POKEMART_MAP_IDS: frozenset[int] = frozenset({
    MAP_VIRIDIAN_MART,
    MAP_PEWTER_MART,
    MAP_CERULEAN_MART,
    MAP_VERMILION_MART,
    MAP_LAVENDER_MART,
    MAP_FUCHSIA_MART,
    MAP_CINNABAR_MART,
    MAP_SAFFRON_MART,
})

# Gym map ids. PEWTER_GYM (Brock) is the V1 target and is verified.
# Other gym ids deferred to V2+ — initial sweep showed my from-memory
# values for Cerulean Gym (0x41) collided with Cerulean Mart (0x41),
# so the rest need empirical verification on the actual ROM before
# being added to the bonus set.
MAP_PEWTER_GYM = 0x36  # Brock (rock) — V1 target

GYM_MAP_IDS: frozenset[int] = frozenset({MAP_PEWTER_GYM})


# --- Readers ---

def read_u8(mem: MemoryView, addr: int) -> int:
    return mem[addr]


def read_u16_be(mem: MemoryView, addr: int) -> int:
    return (mem[addr] << 8) | mem[addr + 1]


def read_bcd(mem: MemoryView, addr: int, length: int) -> int:
    """Read a binary-coded-decimal value. Each nibble is one digit."""
    total = 0
    for i in range(length):
        b = mem[addr + i]
        total = total * 100 + ((b >> 4) * 10) + (b & 0x0F)
    return total


def party_count(mem: MemoryView) -> int:
    return read_u8(mem, ADDR_PARTY_COUNT)


def party_species(mem: MemoryView) -> list[int]:
    n = party_count(mem)
    return [read_u8(mem, ADDR_PARTY_SPECIES + i) for i in range(n)]


def party_levels(mem: MemoryView) -> list[int]:
    n = party_count(mem)
    return [
        read_u8(mem, ADDR_PARTY_MON_BASE + i * PARTY_MON_STRUCT_SIZE + PMON_OFFSET_LEVEL)
        for i in range(n)
    ]


def party_hp(mem: MemoryView) -> list[tuple[int, int]]:
    """Return [(current_hp, max_hp), ...] for each party Pokemon."""
    n = party_count(mem)
    out: list[tuple[int, int]] = []
    for i in range(n):
        base = ADDR_PARTY_MON_BASE + i * PARTY_MON_STRUCT_SIZE
        cur = read_u16_be(mem, base + PMON_OFFSET_HP_CURRENT)
        mx = read_u16_be(mem, base + PMON_OFFSET_HP_MAX)
        out.append((cur, mx))
    return out


def badges_bitfield(mem: MemoryView) -> int:
    return read_u8(mem, ADDR_BADGES)


def badges_count(mem: MemoryView) -> int:
    return bin(badges_bitfield(mem)).count("1")


def map_id(mem: MemoryView) -> int:
    return read_u8(mem, ADDR_MAP_ID)


def player_position(mem: MemoryView) -> tuple[int, int]:
    return read_u8(mem, ADDR_PLAYER_X), read_u8(mem, ADDR_PLAYER_Y)


def money(mem: MemoryView) -> int:
    return read_bcd(mem, ADDR_MONEY, 3)


def in_battle(mem: MemoryView) -> int:
    return read_u8(mem, ADDR_IN_BATTLE)


def enemy_mon_species(mem: MemoryView) -> int:
    return read_u8(mem, ADDR_ENEMY_MON_SPECIES)


def enemy_mon_hp(mem: MemoryView) -> int:
    return read_u16_be(mem, ADDR_ENEMY_MON_HP)


def trainer_id(mem: MemoryView) -> tuple[int, int]:
    """(class, number) — uniquely identifies a trainer in Pokemon Red."""
    return read_u8(mem, ADDR_TRAINER_CLASS), read_u8(mem, ADDR_TRAINER_NUMBER)


def event_flags_popcount(mem: MemoryView) -> int:
    """Total number of set event flags. Useful as a coarse story-progress signal."""
    total = 0
    for i in range(EVENT_FLAGS_BYTES):
        total += bin(mem[ADDR_EVENT_FLAGS_START + i]).count("1")
    return total


def event_flag_bits_set(mem: MemoryView) -> set[int]:
    """Set of global bit-indices currently set in the event-flag region.

    Bit index = byte_offset * 8 + bit_within_byte, range [0, EVENT_FLAGS_BYTES*8).
    Unlike event_flags_popcount (an aggregate that is non-monotone — some
    script/menu bits in this region toggle off and back on), this exposes
    WHICH individual flags are set, so a reward can pay for each newly-set
    flag exactly once and mask it permanently against re-toggling.
    """
    out: set[int] = set()
    for i in range(EVENT_FLAGS_BYTES):
        b = mem[ADDR_EVENT_FLAGS_START + i]
        if b:
            base = i * 8
            for bit in range(8):
                if b & (1 << bit):
                    out.add(base + bit)
    return out


def event_flag_bit(mem: MemoryView, addr: int, bit: int) -> bool:
    """True iff the given event-flag bit is set. `addr`/`bit` come from the
    HARD_GATE_* constants (verified against the pokered disassembly)."""
    return bool(mem[addr] & (1 << bit))


def got_oaks_parcel(mem: MemoryView) -> bool:
    return event_flag_bit(mem, *HARD_GATE_GOT_OAKS_PARCEL)


def got_pokedex(mem: MemoryView) -> bool:
    return event_flag_bit(mem, *HARD_GATE_GOT_POKEDEX)


def beat_brock(mem: MemoryView) -> bool:
    return event_flag_bit(mem, *HARD_GATE_BEAT_BROCK)


def got_pokeballs_from_oak(mem: MemoryView) -> bool:
    return event_flag_bit(mem, *EVENT_GOT_POKEBALLS_FROM_OAK)


def battled_rival_in_oaks_lab(mem: MemoryView) -> bool:
    """True once the opening rival battle is won (the start-state opener)."""
    return event_flag_bit(mem, *EVENT_BATTLED_RIVAL_IN_OAKS_LAB)


def got_potion_sample(mem: MemoryView) -> bool:
    return event_flag_bit(mem, *EVENT_GOT_POTION_SAMPLE)


def got_tm34(mem: MemoryView) -> bool:
    return event_flag_bit(mem, *EVENT_GOT_TM34)


def progress_bits(mem: MemoryView) -> "list[float]":
    """Hidden story-state bits for the RND context vector, in fixed order:
    [oaks_parcel, pokedex, beat_brock, pokeballs_from_oak]. These are the
    invisible-in-pixels flags that make curiosity context-aware (e.g. let it
    tell "Pallet holding the parcel" apart from "Pallet before the parcel")."""
    return [
        float(got_oaks_parcel(mem)),
        float(got_pokedex(mem)),
        float(beat_brock(mem)),
        float(got_pokeballs_from_oak(mem)),
    ]


# Width of progress_bits() — the RND progress-vector dimension.
PROGRESS_DIM = 4


def pokedex_owned_count(mem: MemoryView) -> int:
    """Number of species with their pokedex "owned" bit set. Squirtle is
    already #1 at the curriculum start state, so a fresh agent reads 1."""
    total = 0
    for a in range(ADDR_POKEDEX_OWNED_START, ADDR_POKEDEX_OWNED_END + 1):
        total += bin(mem[a]).count("1")
    return total


def bag_item_quantity(mem: MemoryView, item_id: int) -> int:
    """Quantity of `item_id` in the bag, or 0 if not present. Bag is
    [count][id1, qty1][id2, qty2]...[0xFF terminator]."""
    n = mem[ADDR_BAG_COUNT]
    for i in range(min(n, 20)):
        if mem[ADDR_BAG_ITEMS + 2 * i] == item_id:
            return mem[ADDR_BAG_ITEMS + 2 * i + 1]
    return 0
