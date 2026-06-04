"""Distributed-training helpers (torchrun-driven multi-GPU PPO).

Single-process is the default — when no torchrun env vars are present,
`get_dist_info()` returns rank=0/world_size=1/is_distributed=False and the
trainer runs the original CleanRL-style single-process loop unchanged.

Multi-GPU mode is opt-in via torchrun:
    torchrun --standalone --nproc_per_node=4 -m pokerl.scripts.train ...
Torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def get_dist_info() -> DistContext:
    """Read torchrun env vars (or defaults). No side effects."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=world_size > 1,
    )


def setup(backend: str = "nccl") -> DistContext:
    """Initialize the process group if running under torchrun."""
    dctx = get_dist_info()
    if dctx.is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(dctx.local_rank)
    return dctx


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Cheap helper: average a Python scalar across ranks."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def all_reduce_sum_int(value: int, device: torch.device) -> int:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())
