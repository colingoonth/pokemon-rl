#!/usr/bin/env bash
# Head-to-head: is the PPO UPDATE faster on a GPU than on many CPU threads?
# Settles CPU-node vs GPU-node for single-run PPO throughput. Same loop, same
# envs (16), only the learner device/threads vary. CPU points first so we keep
# that data even if the 1080Ti (Pascal sm_61) has a torch-arch surprise.
#
# Submit: sbatch slurm/bench_1080ti_cpu_vs_gpu.sbatch
#SBATCH --job-name=pkrl_dvbench
#SBATCH --partition=gpu
#SBATCH --constraint=gtx1080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/guenthc1/pokemon-rl/runs/%x_%j.out

set -uo pipefail
cd /scratch/guenthc1/pokemon-rl

# Env workers stay single-threaded (pinned before they spawn). torch_threads in
# the config bumps ONLY the main/learner process for the CPU-update points.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy

echo "host: $(hostname)  cores: $(nproc)  job: ${SLURM_JOB_ID:-local}"
echo "=== CUDA preflight ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2 || echo "nvidia-smi unavailable"
~/.local/bin/uv run python - <<'PY' 2>&1 | grep -viE 'SDL2|UserWarning'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability sm_", "".join(map(str, torch.cuda.get_device_capability(0))), sep="")
    try:
        x = torch.randn(256, 4, 36, 40, device="cuda"); y = (x*2).sum().item()
        print("GPU op OK (sm kernel present), sanity sum:", round(y, 1))
    except Exception as e:
        print("GPU OP FAILED:", repr(e))
PY

BENCH_DIR=runs/bench_dv
CFG_DIR=$BENCH_DIR/configs
mkdir -p "$CFG_DIR"
RESULTS=$BENCH_DIR/dv_results.tsv
echo -e "device\tthreads\tsps\troll_s\tupd_s\twall_s\tstatus" > "$RESULTS"
echo "results -> $RESULTS"

N_ENVS=16
N_STEPS=64
ITERS=6
WARMUP=64

# (device threads) — CPU thread sweep first, then GPU. Same 16 envs throughout.
GRID=( "cpu 1" "cpu 8" "cpu 16" "cuda 1" )

run_point () {
  local device=$1 threads=$2
  local total=$(( N_STEPS * N_ENVS * (ITERS + 1) ))
  local run_name="dv_${device}_t${threads}"
  local cfg="$CFG_DIR/${run_name}.yaml"
  local logf="$BENCH_DIR/${run_name}.log"

  cat > "$cfg" <<EOF
run_name: ${run_name}
state_path: states/blue_fight.state
total_timesteps: ${total}
n_envs: ${N_ENVS}
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

  echo ">>> point device=${device} threads=${threads}"
  local t0=$SECONDS status="ok"
  ~/.local/bin/uv run python -m pokerl.scripts.train --config "$cfg" --device "$device" \
      > "$logf" 2>&1 || status="FAIL"
  local wall=$(( SECONDS - t0 ))

  local line sps roll upd
  line=$(grep -E 'iter +[0-9]+ +step' "$logf" | tail -1)
  if [ -z "$line" ]; then
    sps="NA"; roll="NA"; upd="NA"; [ "$status" = ok ] && status="NO_ITER"
    echo "    (no iter line; tail of log:)"; tail -3 "$logf" | sed 's/^/    /'
  else
    sps=$(echo "$line"  | sed -nE 's/.* sps ([0-9.]+).*/\1/p')
    roll=$(echo "$line" | sed -nE 's/.* roll ([0-9.]+)s.*/\1/p')
    upd=$(echo "$line"  | sed -nE 's/.* upd ([0-9.]+)s.*/\1/p')
  fi
  echo -e "${device}\t${threads}\t${sps}\t${roll}\t${upd}\t${wall}\t${status}" | tee -a "$RESULTS"
}

for point in "${GRID[@]}"; do
  read -r dv th <<< "$point"
  run_point "$dv" "$th"
done

echo "=== DV BENCH COMPLETE ==="
column -t -s $'\t' "$RESULTS"
