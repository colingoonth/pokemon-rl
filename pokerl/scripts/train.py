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


def _snapshot_run(run_dir: Path, cfg, config_src: Path) -> None:
    """Make a run self-describing: write the RESOLVED config, git state, and
    metadata into the run dir at launch (main rank only). Without this a run
    dir is just {metrics.csv, checkpoints/} with no record of what produced it
    — and an uncommitted local edit is exactly how a run becomes unreproducible,
    so the git dirty flag is load-bearing for this edit-config / push / pull /
    sbatch workflow."""
    import json
    import socket
    import subprocess
    from dataclasses import asdict
    from datetime import datetime

    import yaml

    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False)
    )

    def _git(*a: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *a], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    (run_dir / "git.txt").write_text(f"commit {commit}\ndirty {dirty}\n")
    meta = {
        "run_name": run_dir.name,
        "config_src": str(config_src),
        "commit": commit,
        "dirty": dirty,
        "seed": cfg.seed,
        "reward_class": cfg.reward_class,
        "total_timesteps": cfg.total_timesteps,
        "n_envs": cfg.n_envs,
        "torch_threads": cfg.torch_threads,
        "device": cfg.device,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "launched_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


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
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing non-empty run dir. Without this "
                        "(and without --resume) a collision with a run that has "
                        "checkpoints/metrics aborts, to prevent clobbering work.")
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
        # Clobber guard: refuse to write into a dir that already holds a run
        # unless we're resuming it or explicitly forcing. A silent reuse cost a
        # run dir once (Day 10). --resume continues it; --force overwrites.
        ckpt_dir = run_dir / "checkpoints"
        has_run = run_dir.exists() and (
            (run_dir / "metrics.csv").exists()
            or (ckpt_dir.exists() and next(ckpt_dir.glob("iter_*.pt"), None) is not None)
        )
        if has_run and args.resume is None and not args.force:
            raise SystemExit(
                f"run dir already exists with data: {run_dir}\n"
                f"  pass --resume <ckpt> to continue it, --force to overwrite, "
                f"or choose a new run_name in the config."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_csv = str(run_dir / "metrics.csv")
        _snapshot_run(run_dir, cfg, args.config)
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
            max_steps=cfg.max_steps,
            frame_stack=4,
            async_envs=cfg.async_envs,
            reward_cls=reward_cls,
            frame_skip=cfg.frame_skip,
            context=cfg.vec_context,
        )

    try:
        # train() writes run_dir/final.pt as a FULL checkpoint (resumable) on
        # completion; no bare re-save here (that would clobber it).
        train(env_fn, cfg, resume_path=args.resume)
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
