#!/usr/bin/env bash
#SBATCH --job-name=pokerl-v23-n62
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/brock_v0_2_3_n62_%j/train.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/brock_v0_2_3_n62_%j/train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=31
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

RUN_DIR="/scratch/guenthc1/pokemon-rl/runs/brock_v0_2_3_n62_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

cd "/scratch/guenthc1/pokemon-rl"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Training run ${SLURM_JOB_ID} (V0.2.3, n_envs=62) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Workdir: $(pwd)"
echo "Outputs: ${RUN_DIR}"
echo

uv run python -m pokerl.scripts.train --config configs/elsa_brock_v0_2_3_n62.yaml

echo "=== Done $(date) ==="
