#!/usr/bin/env bash
#SBATCH --job-name=pokerl-hello
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/hello_%j/hello.out
#SBATCH --error=/scratch/guenthc1/pokemon-rl/runs/hello_%j/hello.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

# Make sure the output dir exists before SLURM redirects there
mkdir -p "/scratch/guenthc1/pokemon-rl/runs/hello_${SLURM_JOB_ID}"

cd "/scratch/guenthc1/pokemon-rl"

# Activate uv-managed venv
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== ELSA hello-world ==="
echo "Job id:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Workdir:   $(pwd)"
echo
echo "=== GPU info ==="
nvidia-smi || echo "nvidia-smi not available"
echo
echo "=== Python / Torch ==="
uv run python - <<'EOF'
import torch
print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    x = torch.zeros(1024, 1024, device="cuda")
    y = (x + 1).sum().item()
    print(f"CUDA tensor smoke: sum = {y}  (expected ~{1024 * 1024})")
EOF

echo
echo "=== Done ==="
