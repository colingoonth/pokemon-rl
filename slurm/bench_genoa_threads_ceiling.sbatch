#!/usr/bin/env bash
# Find where multi-threaded PPO-update SGD SATURATES on EPYC Genoa. The sps
# ceiling is ~ (threaded SGD throughput / n_epochs), so if the update keeps
# scaling past 32 threads, the whole run goes faster. The 1080Ti's slow Skylake
# cores barely improved 8->16; EPYC has far more memory bandwidth and may scale
# further. Runs in PARALLEL with the n_envs x threads sweep on a separate node.
#
# Submit: sbatch slurm/bench_genoa_threads_ceiling.sbatch
#SBATCH --job-name=pkrl_tceil
#SBATCH --partition=long
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=190
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/%x_%j.out

set -uo pipefail
cd /scratch/guenthc1/pokemon-rl

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy

echo "host: $(hostname)  cores: $(nproc)  job: ${SLURM_JOB_ID:-local}"

BENCH_DIR=runs/bench_tceil
CFG_DIR=$BENCH_DIR/configs
mkdir -p "$CFG_DIR"
RESULTS=$BENCH_DIR/tceil_results.tsv
echo -e "n_envs\tthreads\tsps\troll_s\tupd_s\twall_s\tstatus" > "$RESULTS"
echo "results -> $RESULTS"

N_STEPS=64
ITERS=6
WARMUP=64

# Push threads high at two env counts to locate the SGD-threading knee.
GRID=(
  "64 16"  "64 32"  "64 48"  "64 64"  "64 96"
  "128 32" "128 64" "128 96"
)

run_point () {
  local n_envs=$1 threads=$2
  local total=$(( N_STEPS * n_envs * (ITERS + 1) ))
  local run_name="tceil_e${n_envs}_t${threads}"
  local cfg="$CFG_DIR/${run_name}.yaml"
  local logf="$BENCH_DIR/${run_name}.log"

  cat > "$cfg" <<EOF
run_name: ${run_name}
state_path: states/blue_fight.state
total_timesteps: ${total}
n_envs: ${n_envs}
max_steps: 32768
n_steps: ${N_STEPS}
n_epochs: 4
minibatch_size: 64
learning_rate: 2.5e-4
gamma: 0.999
gae_lambda: 0.95
clip_coef: 0.1
value_coef: 0.5
entropy_coef: 0.01
max_grad_norm: 0.5
anneal_lr: true
seed: 0
log_every: 1
save_every: 0
async_envs: true
reward_class: RewardV0_5_skeleton
frame_skip: 1
torch_threads: ${threads}
vec_context: spawn
rnd_enabled: true
ext_coef: 2.0
int_coef: 1.0
int_gamma: 0.99
rnd_update_proportion: 0.25
rnd_feature_dim: 256
rnd_obs_norm_steps: ${WARMUP}
rnd_obs_clip: 5.0
rnd_int_clip: 5.0
EOF

  echo ">>> point n_envs=${n_envs} threads=${threads} (total=${total})"
  local t0=$SECONDS status="ok"
  ~/.local/bin/uv run python -m pokerl.scripts.train --config "$cfg" --device cpu \
      > "$logf" 2>&1 || status="FAIL"
  local wall=$(( SECONDS - t0 ))

  local line sps roll upd
  line=$(grep -E 'iter +[0-9]+ +step' "$logf" | tail -1)
  if [ -z "$line" ]; then
    sps="NA"; roll="NA"; upd="NA"; [ "$status" = ok ] && status="NO_ITER"
    tail -3 "$logf" | sed 's/^/    /'
  else
    sps=$(echo "$line"  | sed -nE 's/.* sps ([0-9.]+).*/\1/p')
    roll=$(echo "$line" | sed -nE 's/.* roll ([0-9.]+)s.*/\1/p')
    upd=$(echo "$line"  | sed -nE 's/.* upd ([0-9.]+)s.*/\1/p')
  fi
  echo -e "${n_envs}\t${threads}\t${sps}\t${roll}\t${upd}\t${wall}\t${status}" | tee -a "$RESULTS"
}

for point in "${GRID[@]}"; do
  read -r ne th <<< "$point"
  run_point "$ne" "$th"
done

echo "=== THREADS-CEILING SWEEP COMPLETE ==="
column -t -s $'\t' "$RESULTS"
