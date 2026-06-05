#!/usr/bin/env bash
#SBATCH --job-name=pokerl-v42-c-ddp3
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v0_4_2_center_ddp3_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v0_4_2_center_ddp3_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v0_4_2_center_ddp3_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# Thread isolation — each PyBoy subprocess gets one thread of numpy/blas
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NCCL hygiene
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Training run ${SLURM_JOB_ID} (V0.4.2 center DDP, 3xL40S) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  -m pokerl.scripts.train \
  --config configs/elsa_brock_v0_4_2_center_ddp3.yaml

echo "=== Done $(date) ==="
