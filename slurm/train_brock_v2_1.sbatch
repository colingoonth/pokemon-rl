#!/usr/bin/env bash
#SBATCH --job-name=pokerl-brock-v2-1
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v2_1_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v2_1_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v2_1_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Training run ${SLURM_JOB_ID} (V2.1) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
echo

uv run python -m pokerl.scripts.train --config configs/elsa_brock_v2_1.yaml

echo "=== Done $(date) ==="
