#!/usr/bin/env bash
#SBATCH --job-name=pokerl-sanity-2n
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/sanity_2n_%j/out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/sanity_2n_%j/err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40s
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

mkdir -p "/scratch/guenthc1/pokemon-rl/runs/sanity_2n_${SLURM_JOB_ID}"
cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# NCCL for multi-node Ethernet (no Infiniband on ELSA L40S nodes)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=INFO         # verbose for first multi-node run
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

# Rendezvous: first node = master; per-job port to avoid collisions with
# other multi-node jobs landing on overlapping nodes.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "=== Sanity 2N (rendezvous + all_reduce) — job ${SLURM_JOB_ID} ==="
echo "Nodes:  $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Date:   $(date)"

srun --kill-on-bad-exit=1 bash -c '
uv run torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --max-restarts=0 \
  -m pokerl.scripts.bench_nccl
'

echo "=== Done $(date) ==="
