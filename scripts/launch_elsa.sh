#!/usr/bin/env bash
# Fire off a Genoa training batch on ELSA from one command.
# Parses run_name from the config, syncs the config up, and submits a
# fully-formed threaded Genoa job (190 cores, OMP pinned, torch_threads from the
# config). No per-run sbatch hand-editing.
#
#   ./scripts/launch_elsa.sh configs/elsa_cpu_genoa_h32k.yaml
#   ./scripts/launch_elsa.sh configs/foo.yaml --partition nolimit --time 30-00:00:00
#   ./scripts/launch_elsa.sh configs/foo.yaml --resume   # continue from latest checkpoint
#   ./scripts/launch_elsa.sh configs/foo.yaml --force     # overwrite an existing run dir
#
# NOTE: this syncs the CONFIG, not code. If you changed pokerl/ or the launcher,
# rsync the repo to ELSA first (the train.py clobber-guard etc. must be current).
set -euo pipefail

HOST="guenthc1@elsa.hpc.tcnj.edu"
REPO="/scratch/guenthc1/pokemon-rl"
PARTITION="long"
TIME="7-00:00:00"
CPUS=190
MEM="200G"
RESUME=0
FORCE=0
CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2;;
    --time)      TIME="$2"; shift 2;;
    --cpus)      CPUS="$2"; shift 2;;
    --mem)       MEM="$2"; shift 2;;
    --resume)    RESUME=1; shift;;
    --force)     FORCE=1; shift;;
    -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    -*)          echo "unknown flag: $1" >&2; exit 1;;
    *)           CONFIG="$1"; shift;;
  esac
done

[[ -n "$CONFIG" ]] || { echo "usage: launch_elsa.sh <config.yaml> [--partition P] [--time T] [--resume] [--force]" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

RUN_NAME=$(grep -E '^run_name:' "$CONFIG" | head -1 | sed -E 's/^run_name:[[:space:]]*//; s/[[:space:]]*$//')
[[ -n "$RUN_NAME" ]] || { echo "no run_name: in $CONFIG" >&2; exit 1; }
JOB="pkrl_${RUN_NAME}"
CONFIG_REL="${CONFIG#./}"
EXTRA=""
[[ "$FORCE" = 1 ]] && EXTRA="--force"

echo "run_name : $RUN_NAME"
echo "config   : $CONFIG_REL"
echo "partition: $PARTITION   time: $TIME   cpus: $CPUS   mem: $MEM"
echo "resume   : $RESUME   force: $FORCE"

# Sync the config up (relative path preserved so it lands at the same repo path).
rsync -aR "$CONFIG_REL" "$HOST:$REPO/" >/dev/null

# Submit: pipe a generated sbatch to `sbatch` on the login node. Local vars
# expand here; \$-escaped ones (LATEST, hostname, SLURM_JOB_ID) run on the node.
ssh -o BatchMode=yes "$HOST" sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB}
#SBATCH --partition=${PARTITION}
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${REPO}/runs/%x_%j.out
set -euo pipefail
cd ${REPO}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
RESUME_ARG=""
if [ "${RESUME}" = 1 ]; then
  LATEST=\$(ls -1 ${REPO}/runs/${RUN_NAME}/checkpoints/iter_*.pt 2>/dev/null | sort | tail -1 || true)
  if [ -n "\$LATEST" ]; then RESUME_ARG="--resume \$LATEST"; echo "resuming from \$LATEST"; else echo "no checkpoint to resume; starting fresh"; fi
fi
echo "host: \$(hostname)  cores: \$(nproc)  job: \$SLURM_JOB_ID  run: ${RUN_NAME}"
~/.local/bin/uv run python -m pokerl.scripts.train --config ${CONFIG_REL} --device cpu ${EXTRA} \$RESUME_ARG
EOF

echo "submitted. watch with: uv run python -m pokerl.scripts.status ${RUN_NAME} --remote   (or squeue -u guenthc1)"
