#!/usr/bin/env bash
#SBATCH --job-name=pokerl-v42-h-ddp2
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v0_4_2_harsh_ddp2_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v0_4_2_harsh_ddp2_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v0_4_2_harsh_ddp2_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Training run ${SLURM_JOB_ID} (V0.4.2 harsh DDP, 2xL40S) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  -m pokerl.scripts.train \
  --config configs/elsa_brock_v0_4_2_harsh_ddp2.yaml

echo "=== Done $(date) ==="
