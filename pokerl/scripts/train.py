"""PPO training entry point.

Single-process usage:
  uv run python -m pokerl.scripts.train --config configs/elsa_brock.yaml

Multi-GPU DDP (single-node) usage — torchrun owns rendezvous:
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
    -m pokerl.scripts.train --config configs/elsa_brock_v0_2_3_ddp4.yaml

Run artifacts (csv log, checkpoint) land in runs/<run_name>/.
"""
from __future__ import annotations

# Thread-isolation env vars MUST be set before importing torch — torch reads
# OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS at import time and
# spawned env-worker subprocesses inherit os.environ from the parent. Without
# this, each PyBoy subprocess tries to fan its internal numpy/blas calls
# across all CPUs and fights the other ranks for cores.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path

import torch

from pokerl.agent.ppo import train
from pokerl.env.make import make_vec_env
from pokerl.env.rewards import get_reward_cls
from pokerl.infra import dist as dist_helpers
from pokerl.infra.config import load_ppo_config

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "dev_local.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--device", default=None,
                   help="Override device (cpu / cuda). Default: auto.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Warm-start: load this checkpoint into the policy "
                        "before training begins. Useful when continuing an "
                        "earlier run; harmful when the earlier run learned "
                        "a degenerate policy you want to discard.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Init process group BEFORE choosing device, so torch.cuda.set_device runs
    # for the right local_rank in dist_helpers.setup.
    dctx = dist_helpers.setup()

    if args.device is not None:
        device = args.device
    elif dctx.is_distributed:
        device = f"cuda:{dctx.local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = ROOT / "runs"
    cfg, run_name = load_ppo_config(args.config, device=device)
    run_dir = out_root / run_name
    if dctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_csv = str(run_dir / "metrics.csv")
    else:
        # Non-main ranks must NOT write CSV/checkpoints. ppo.train() already
        # guards on dctx.is_main but blanking the path defensively avoids
        # accidental rank-N writes if logic ever changes.
        cfg.log_csv = None

    if dctx.is_main:
        print(f"Config: {args.config}")
        print(f"Run:    {run_name}")
        print(f"Device: {device}")
        print(f"World:  {dctx.world_size} (rank {dctx.rank}, local {dctx.local_rank})")
        print(f"PPO:    {cfg}")

    reward_cls = get_reward_cls(cfg.reward_class)
    if dctx.is_main:
        print(f"Reward: {cfg.reward_class}")

    # Per-rank seed offset. Single-node has only rank 0, so behavior is
    # unchanged. Multi-rank: each rank rolls a different env trajectory,
    # so 32 ranks aren't running 32 identical worlds.
    cfg.seed = cfg.seed + dctx.rank * 1000

    # Resolve relative state_path against repo root before passing into env_fn.
    # Async env subprocesses are spawned with their own cwd inheritance; under
    # multi-node DDP each rank may launch from a node where cwd handling is
    # less reliable. Absolute paths sidestep this class of failure.
    state_path = cfg.state_path
    if state_path is not None:
        sp = Path(state_path)
        if not sp.is_absolute():
            sp = ROOT / sp
        state_path = str(sp)

    n_envs_per_rank = cfg.n_envs // dctx.world_size

    def env_fn():
        return make_vec_env(
            n_envs=n_envs_per_rank,
            state_path=state_path,
            headless=True,
            max_steps=4096,
            frame_stack=4,
            async_envs=cfg.async_envs,
            reward_cls=reward_cls,
            frame_skip=cfg.frame_skip,
        )

    try:
        net = train(env_fn, cfg, resume_path=args.resume)
        if dctx.is_main:
            torch.save(net.state_dict(), run_dir / "final.pt")
            print(f"Saved final checkpoint -> {run_dir / 'final.pt'}")
    finally:
        # Drain any in-flight collectives before tearing down the process
        # group, then destroy. envs are already closed inside train();
        # barrier here only matters when distributed.
        if dctx.is_distributed:
            import torch.distributed as _d
            if _d.is_initialized():
                _d.barrier()
        dist_helpers.cleanup()


if __name__ == "__main__":
    main()
