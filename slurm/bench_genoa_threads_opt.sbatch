#!/usr/bin/env bash
# Optimize (n_envs x torch_threads) on a full Genoa node to MAX single-run sps.
# The 1080Ti head-to-head proved multi-threading the PPO update is a ~4x win and
# flips the bottleneck back to rollout. On 190 fast EPYC cores we can afford both
# generous update threads AND many envs (they timeshare, rollout then update).
# Goal: find the peak and lock the production config.
#
# Submit: sbatch slurm/bench_genoa_threads_opt.sbatch
#SBATCH --job-name=pkrl_topt
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

# Workers single-threaded (pinned before spawn); torch_threads bumps ONLY the
# main/learner process AFTER workers exist.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy

echo "host: $(hostname)  cores: $(nproc)  job: ${SLURM_JOB_ID:-local}"

BENCH_DIR=runs/bench_topt
CFG_DIR=$BENCH_DIR/configs
mkdir -p "$CFG_DIR"
RESULTS=$BENCH_DIR/topt_results.tsv
echo -e "n_envs\tthreads\tsps\troll_s\tupd_s\twall_s\tstatus" > "$RESULTS"
echo "results -> $RESULTS"

N_STEPS=64
ITERS=6
WARMUP=64

# (n_envs threads) — explore the rollout/update balance on 190 cores.
GRID=(
  "32 8"  "32 16"
  "64 8"  "64 16" "64 32"
  "128 16" "128 32"
  "160 16" "160 32"
)

run_point () {
  local n_envs=$1 threads=$2
  local total=$(( N_STEPS * n_envs * (ITERS + 1) ))
  local run_name="topt_e${n_envs}_t${threads}"
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

echo "=== THREADS-OPT SWEEP COMPLETE ==="
column -t -s $'\t' "$RESULTS"
