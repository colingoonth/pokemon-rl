#!/usr/bin/env bash
# Sync run artifacts (checkpoints, logs, eval gifs) from ELSA's scratch
# back to local. Idempotent: rsync only pulls what changed.
#
# Usage:
#   ./scripts/sync_from_elsa.sh            # sync everything under runs/
#   ./scripts/sync_from_elsa.sh brock_1234 # sync one specific run
set -euo pipefail

REMOTE_USER="guenthc1"
REMOTE_HOST="elsa.hpc.tcnj.edu"
REMOTE_BASE="/scratch/${REMOTE_USER}/pokemon-rl/runs"
LOCAL_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs"

mkdir -p "${LOCAL_BASE}"

if [[ $# -eq 1 ]]; then
    SRC="${REMOTE_BASE}/${1}/"
    DST="${LOCAL_BASE}/${1}/"
    mkdir -p "${DST}"
    echo "Syncing ${REMOTE_HOST}:${SRC} -> ${DST}"
    rsync -avh --progress "${REMOTE_USER}@${REMOTE_HOST}:${SRC}" "${DST}"
else
    echo "Syncing ${REMOTE_HOST}:${REMOTE_BASE}/ -> ${LOCAL_BASE}/"
    rsync -avh --progress "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/" "${LOCAL_BASE}/"
fi
