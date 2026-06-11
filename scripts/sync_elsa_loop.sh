#!/usr/bin/env bash
# Poll an ELSA run's checkpoint dir and pull the latest checkpoint to the Mac
# for local watch.py / rollout.py eval. Key-based SSH (no password, no secrets).
# Usage: ./scripts/sync_elsa_loop.sh <run_name> [interval_seconds]
set -u
RUN="${1:?run_name}"
INTERVAL="${2:-600}"
HOST=guenthc1@elsa.hpc.tcnj.edu
REMOTE_DIR="/scratch/guenthc1/pokemon-rl/runs/$RUN/checkpoints"
DST="/Users/colin/pokemon-rl/checkpoints/cmp/${RUN}_latest.pt"
LOG="/Users/colin/pokemon-rl/checkpoints/cmp/sync_${RUN}.log"
mkdir -p "$(dirname "$DST")"
while true; do
  latest=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "ls -t $REMOTE_DIR/iter_*.pt 2>/dev/null | head -1" 2>/dev/null | tr -d '\r')
  if [ -n "$latest" ]; then
    if rsync -az -e "ssh -o BatchMode=yes" "$HOST:$latest" "$DST" 2>/dev/null; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] synced $(basename "$latest") -> ${RUN}_latest.pt" >> "$LOG"
    fi
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] no checkpoint yet" >> "$LOG"
  fi
  sleep "$INTERVAL"
done
