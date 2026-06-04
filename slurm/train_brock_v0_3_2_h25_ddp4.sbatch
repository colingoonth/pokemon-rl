#!/usr/bin/env bash
#SBATCH --job-name=pokerl-v32-h25-ddp4
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v0_3_2_h25_ddp4_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v0_3_2_h25_ddp4_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=31
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v0_3_2_h25_ddp4_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# Thread isolation — each PyBoy subprocess gets one thread of numpy/blas,
# so 32 subprocesses (4 ranks * 8 envs) don't fight for cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NCCL hygiene: warn on real issues, fast-fail on hangs.
# Both var names — modern (TORCH_*) is the PyTorch 2.2+ canonical;
# legacy is kept for forward/back compat.
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Training run ${SLURM_JOB_ID} (V0.3.2 h25 DDP, 4xL40S) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# Single-node DDP via torchrun. --standalone uses a free local port (no
# MASTER_ADDR/PORT plumbing needed for single-node). Do NOT wrap with srun
# here — the sbatch is already a single-task allocation (--ntasks=1) and
# the `srun torchrun ...` double-launch is a known hang.
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m pokerl.scripts.train \
  --config configs/elsa_brock_v0_3_2_h25_ddp4.yaml

echo "=== Done $(date) ==="
