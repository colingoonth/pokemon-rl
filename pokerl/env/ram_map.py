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

# Event flags: a long bitarray. Each story flag is one bit somewhere in here.
ADDR_EVENT_FLAGS_START = 0xD747
ADDR_EVENT_FLAGS_END   = 0xD886  # inclusive
EVENT_FLAGS_BYTES      = ADDR_EVENT_FLAGS_END - ADDR_EVENT_FLAGS_START + 1

# Known map ids (handful — full list in pret/pokered constants/map_constants.asm)
MAP_PALLET_TOWN        = 0x00
MAP_VIRIDIAN_CITY      = 0x01
MAP_PEWTER_CITY        = 0x02
MAP_ROUTE_1            = 0x0C
MAP_OAKS_LAB           = 0x28


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


def event_flags_popcount(mem: MemoryView) -> int:
    """Total number of set event flags. Useful as a coarse story-progress signal."""
    total = 0
    for i in range(EVENT_FLAGS_BYTES):
        total += bin(mem[ADDR_EVENT_FLAGS_START + i]).count("1")
    return total
