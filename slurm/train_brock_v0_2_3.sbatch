#!/usr/bin/env bash
#SBATCH --job-name=pokerl-brock-v0-2-3
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v0_2_3_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v0_2_3_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=31
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v0_2_3_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Training run ${SLURM_JOB_ID} (V0.2.3) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
echo

uv run python -m pokerl.scripts.train --config configs/elsa_brock_v0_2_3.yaml

echo "=== Done $(date) ==="
