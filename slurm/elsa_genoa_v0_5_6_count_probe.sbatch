#!/usr/bin/env bash
# V0.5.6 count-novelty PROBE — warm-start from the 100M V0.5.5 checkpoint, on a
# single 192-core Genoa CPU node (ELSA). Submit: sbatch slurm/elsa_genoa_v0_5_6_count_probe.sbatch
#
# SINGLE-PROCESS ONLY (no DDP / multi-GPU): count-novelty reuses the RND
# intrinsic path, which ppo.py guards as single-process (raises under DDP). Like
# the rest of the V0.5 storyladder line, this runs on ONE Genoa CPU node with
# process-level env parallelism. There is no ddp3/ddp4 variant for this run.
#SBATCH --job-name=pkrl_v056_count_probe
#SBATCH --partition=long
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=190
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/%x_%j.out

set -euo pipefail
cd /scratch/guenthc1/pokemon-rl

# CRITICAL: pin per-process BLAS threading to 1. Without this each of the 144
# env subprocesses spawns ~192 OpenMP/MKL threads and the node thrashes. The
# throughput here comes from process-level env parallelism, not intra-op threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy   # headless PyBoy

echo "host: $(hostname)  cores: $(nproc)  job: $SLURM_JOB_ID"

# --- Weights-only warm-start seed --------------------------------------------
# The 100M V0.5.5 checkpoint is a FULL dict {net, rnd, optimizer, ...}. For a
# clean warm-start into COUNT mode we want the POLICY WEIGHTS ONLY (fresh
# optimizer / LR / step counter, and NO RND state — we are switching intrinsic
# modes). Extract the bare `net` state_dict to a seed file; ppo.py auto-detects a
# bare state_dict on --resume and does a weights-only transfer (strict=False, so
# the critic_int head just carries over). A FULL-resume would instead crash in
# count mode (no rnd_model to restore) and would trip the "resume_iteration >=
# n_iterations" guard anyway (100M ckpt into a 30M run), so weights-only is the
# only correct path.
#
# O: ensure SEED_FULL exists on ELSA (rsync v5_1_warmstart_final.pt there). Path
# below is the assumed location — adjust if you place it elsewhere.
SEED_FULL=/scratch/guenthc1/pokemon-rl/checkpoints/v5_1_warmstart_final.pt
SEED_BARE=/scratch/guenthc1/pokemon-rl/checkpoints/v0_5_6_count_seed_net.pt

mkdir -p "$(dirname "$SEED_BARE")"
~/.local/bin/uv run python -c "
import sys, torch
d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
net = d['net'] if isinstance(d, dict) and 'net' in d else d
torch.save(net, sys.argv[2])
print(f'extracted bare net weights -> {sys.argv[2]} ({len(net)} tensors)')
" "$SEED_FULL" "$SEED_BARE"

~/.local/bin/uv run python -m pokerl.scripts.train \
    --config configs/elsa_cpu_genoa_v0_5_6_count_probe.yaml \
    --device cpu \
    --resume "$SEED_BARE"
