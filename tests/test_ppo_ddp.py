"""Tests for the distributed-training plumbing.

Multi-rank coverage requires real GPUs + torchrun, so we don't exercise
the DDP path itself in unit tests. We DO cover:
  - the degenerate world_size=1 path (env vars unset)
  - DistContext parsing from torchrun env vars
  - the rank-0 / non-main IO gating logic indirectly via dist_helpers
"""
from __future__ import annotations

import os
from unittest.mock import patch

from pokerl.infra import dist as dist_helpers


def test_dist_info_defaults_to_single_process():
    """No torchrun env vars -> world_size=1, is_distributed=False."""
    with patch.dict(os.environ, {}, clear=False):
        # Defensive: scrub any leftover torchrun vars from the test env
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.pop(k, None)
        ctx = dist_helpers.get_dist_info()
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.is_distributed is False
    assert ctx.is_main is True


def test_dist_info_parses_torchrun_env_vars():
    """RANK=2, WORLD_SIZE=4, LOCAL_RANK=2 -> distributed context."""
    env = {"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"}
    with patch.dict(os.environ, env):
        ctx = dist_helpers.get_dist_info()
    assert ctx.world_size == 4
    assert ctx.rank == 2
    assert ctx.local_rank == 2
    assert ctx.is_distributed is True
    assert ctx.is_main is False


def test_dist_info_rank_0_of_4_is_main():
    env = {"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0"}
    with patch.dict(os.environ, env):
        ctx = dist_helpers.get_dist_info()
    assert ctx.is_main is True
    assert ctx.is_distributed is True


def test_all_reduce_helpers_passthrough_in_single_process():
    """When dist isn't initialized, helpers return values unchanged."""
    import torch
    device = torch.device("cpu")
    assert dist_helpers.all_reduce_mean(42.0, device) == 42.0
    assert dist_helpers.all_reduce_sum_int(7, device) == 7
