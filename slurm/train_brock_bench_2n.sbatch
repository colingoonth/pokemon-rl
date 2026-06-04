#!/usr/bin/env bash
#SBATCH --job-name=pokerl-bench-2n
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/bench_2n_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/bench_2n_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=31
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/bench_2n_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# Thread isolation — each PyBoy subprocess gets one thread of numpy/blas,
# so 24 subprocesses/node x 2 nodes don't fight for cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NCCL for multi-node Ethernet (no Infiniband on ELSA L40S nodes —
# without these NCCL probes for IB, hangs, then falls back).
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=INFO         # verbose for first throughput run
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600    # catch silent node death within 10min

# Rendezvous: first node = master; per-job port to avoid collisions.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "=== Bench 2N (V0.3.1 reward, 48 envs) — job ${SLURM_JOB_ID} ==="
echo "Nodes:   $SLURM_JOB_NODELIST"
echo "Master:  $MASTER_ADDR:$MASTER_PORT"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# One srun task per node; torchrun fans out to 4 GPUs locally.
srun --kill-on-bad-exit=1 bash -c '
uv run torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --max-restarts=0 \
  -m pokerl.scripts.train \
  --config configs/elsa_brock_bench_2n.yaml
'

echo "=== Done $(date) ==="
