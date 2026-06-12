#!/usr/bin/env bash
# V0.5 RND long-horizon cold run on a 192-core Genoa CPU node (ELSA).
# Submit: sbatch slurm/elsa_genoa_cpu_h32k.sbatch
#SBATCH --job-name=pkrl_genoa_h32k
#SBATCH --partition=long
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=190
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/%x_%j.out

set -euo pipefail
cd /scratch/guenthc1/pokemon-rl

# CRITICAL: pin per-process BLAS threading to 1. Without this each of the 144
# env subprocesses spawns ~192 OpenMP/MKL threads and the node thrashes. The
# throughput here comes from process-level env parallelism, not intra-op threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy   # headless PyBoy

echo "host: $(hostname)  cores: $(nproc)  job: $SLURM_JOB_ID"
~/.local/bin/uv run python -m pokerl.scripts.train \
    --config configs/elsa_cpu_genoa_h32k.yaml --device cpu
