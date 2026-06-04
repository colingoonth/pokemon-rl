#!/usr/bin/env bash
#SBATCH --job-name=pokerl-v32-h10-1gpu
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v0_3_2_h10_1gpu_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v0_3_2_h10_1gpu_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v0_3_2_h10_1gpu_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# Thread isolation — 8 PyBoy subprocesses share 8 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== Training run ${SLURM_JOB_ID} (V0.3.2 h10, 1xL40S) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# Single-GPU: plain python, no torchrun. dist_helpers.setup() detects
# the absence of WORLD_SIZE env var and runs the non-distributed loop.
uv run python -m pokerl.scripts.train \
  --config configs/elsa_brock_v0_3_2_h10_1gpu.yaml

echo "=== Done $(date) ==="
