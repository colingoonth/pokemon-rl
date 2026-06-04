"""Multi-node NCCL rendezvous + all-reduce sanity check.

Launched via the same `srun bash -c 'torchrun ...'` pattern as training.
Verifies (a) all ranks rendezvous, (b) collective ops work across nodes,
(c) reports effective bandwidth on a 10MB tensor.

Usage (via sbatch):
  sbatch slurm/sanity_multinode.sbatch
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"world_size={world_size}, master="
              f"{os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")

    dist.barrier()
    print(f"rank {rank} (local {local_rank}) on {os.uname().nodename}: alive")
    dist.barrier()

    n = 2_500_000  # 10MB float32
    t = torch.ones(n, dtype=torch.float32, device=f"cuda:{local_rank}")

    for _ in range(3):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iters = 20
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    bytes_per_iter = n * 4
    bandwidth_GB_s = (bytes_per_iter * iters) / elapsed / 1e9
    if rank == 0:
        print(f"all_reduce 10MB x {iters}: {elapsed:.3f}s "
              f"-> {bandwidth_GB_s:.3f} GB/s effective")
        print("rendezvous + collective OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
