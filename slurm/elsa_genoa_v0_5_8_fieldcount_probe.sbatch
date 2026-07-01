#!/usr/bin/env bash
# V0.5.8 within-map field refinement PROBE (full-LR) — warm-start from the 100M V0.5.5
# checkpoint, single Genoa CPU node (ELSA).
# Submit: sbatch slurm/elsa_genoa_v0_5_8_fieldcount_probe.sbatch
#
# SINGLE-PROCESS ONLY (no DDP): count-novelty reuses the RND intrinsic path, which
# ppo.py guards as single-process. One Genoa CPU node, process-level env parallelism.
#SBATCH --job-name=pkrl_v058_fieldcount_probe
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

# Pin per-process BLAS threading to 1 — throughput is process-level env parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy   # headless PyBoy

echo "host: $(hostname)  cores: $(nproc)  job: $SLURM_JOB_ID"

# --- Weights-only warm-start seed (same 100M V0.5.5 checkpoint as V0.5.6) --------
# Extract the bare `net` state_dict; ppo.py auto-detects it on --resume and does a
# weights-only transfer (fresh optimizer / LR / step, NO RND state). The new
# critic_int head carries over; the added DIR_FIELD reward is an env-side change,
# invisible to the warm-start (no new net params).
SEED_FULL=/scratch/guenthc1/pokemon-rl/checkpoints/v5_1_warmstart_final.pt
SEED_BARE=/scratch/guenthc1/pokemon-rl/checkpoints/v0_5_8_fieldcount_seed_net.pt

mkdir -p "$(dirname "$SEED_BARE")"
~/.local/bin/uv run python -c "
import sys, torch
d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
net = d['net'] if isinstance(d, dict) and 'net' in d else d
torch.save(net, sys.argv[2])
print(f'extracted bare net weights -> {sys.argv[2]} ({len(net)} tensors)')
" "$SEED_FULL" "$SEED_BARE"

~/.local/bin/uv run python -m pokerl.scripts.train \
    --config configs/elsa_cpu_genoa_v0_5_8_fieldcount_probe.yaml \
    --device cpu \
    --resume "$SEED_BARE"
