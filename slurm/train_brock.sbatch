#!/usr/bin/env bash
#SBATCH --job-name=pokerl-brock
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"

echo "=== Training run ${SLURM_JOB_ID} ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
echo

# Config-driven run. Override checkpoint dir to land in the run-specific path.
uv run python -m pokerl.scripts.train

echo "=== Done $(date) ==="
