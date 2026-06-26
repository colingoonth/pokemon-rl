#!/usr/bin/env bash
# Day 10 throughput sweep on a single 192-core Genoa CPU node (ELSA).
# Goal: find the n_envs knee + which lever (torch_threads, fork vs spawn) moves
# steady-state sps, to fix the single-process coordinator bottleneck that pinned
# the real run (job 22382) at 228 sps while using ~2 of 192 cores.
#
# Each grid point runs a SHORT real PPO loop (same code path as production) for
# ~6 iters and we read the last iter's `sps` + the roll/upd split. frame_skip is
# held at 1 (a learning-semantics knob, swept separately later, not here).
#
# Submit: sbatch slurm/bench_genoa_sweep.sbatch
#SBATCH --job-name=pkrl_bench
#SBATCH --partition=long
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=176
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/%x_%j.out

set -uo pipefail
cd /scratch/guenthc1/pokemon-rl

# Workers MUST stay single-threaded (pinned before they spawn). torch_threads
# in the config bumps ONLY the main process, after the workers exist.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy

BENCH_DIR=runs/bench
CFG_DIR=$BENCH_DIR/configs
mkdir -p "$CFG_DIR"
RESULTS=$BENCH_DIR/sweep_results.tsv
echo -e "n_envs\tthreads\tctx\tsps\troll_s\tupd_s\twall_s\tstatus" > "$RESULTS"

echo "host: $(hostname)  cores: $(nproc)  job: ${SLURM_JOB_ID:-local}"
echo "results -> $RESULTS"

# Short-run knobs shared by every grid point.
N_STEPS=64          # PPO buffer length; sps is per-step-normalized so this is just speed
ITERS=6             # training iters per point (read the last one as steady state)
WARMUP=64           # RND obs-norm warmup steps (real run uses 1024; tiny here to save wall)

# Grid: "n_envs threads context". Phase A = n_envs knee; B = threads at the knee;
# C = fork vs spawn at a few points.
GRID=(
  # --- A: n_envs knee (threads=1, spawn) ---
  "16 1 spawn" "32 1 spawn" "48 1 spawn" "64 1 spawn" "96 1 spawn" "128 1 spawn" "144 1 spawn"
  # --- B: main-process threads at n_envs=64 (spawn) ---
  "64 2 spawn" "64 4 spawn" "64 8 spawn" "64 16 spawn"
  # --- C: fork vs spawn ---
  "64 1 fork" "64 4 fork" "96 1 fork" "144 1 fork"
)

run_point () {
  local n_envs=$1 threads=$2 ctx=$3
  local total=$(( N_STEPS * n_envs * (ITERS + 1) ))
  local run_name="bench_e${n_envs}_t${threads}_${ctx}"
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
vec_context: ${ctx}
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

  echo ">>> point n_envs=${n_envs} threads=${threads} ctx=${ctx} (total=${total})"
  local t0=$SECONDS
  local status="ok"
  ~/.local/bin/uv run python -m pokerl.scripts.train --config "$cfg" --device cpu \
      > "$logf" 2>&1 || status="FAIL"
  local wall=$(( SECONDS - t0 ))

  local line sps roll upd
  line=$(grep -E 'iter +[0-9]+ +step' "$logf" | tail -1)
  if [ -z "$line" ]; then
    sps="NA"; roll="NA"; upd="NA"; [ "$status" = ok ] && status="NO_ITER"
  else
    sps=$(echo "$line"  | sed -nE 's/.* sps ([0-9.]+).*/\1/p')
    roll=$(echo "$line" | sed -nE 's/.* roll ([0-9.]+)s.*/\1/p')
    upd=$(echo "$line"  | sed -nE 's/.* upd ([0-9.]+)s.*/\1/p')
  fi
  echo -e "${n_envs}\t${threads}\t${ctx}\t${sps}\t${roll}\t${upd}\t${wall}\t${status}" | tee -a "$RESULTS"
}

for point in "${GRID[@]}"; do
  read -r ne th cx <<< "$point"
  run_point "$ne" "$th" "$cx"
done

echo "=== SWEEP COMPLETE ==="
column -t -s $'\t' "$RESULTS"
