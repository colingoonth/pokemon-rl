#!/usr/bin/env bash
# Launch a WARM-START training run on VCL in a persistent tmux session.
# Loads weights from <resume_checkpoint> into the policy, then trains with
# <config> (which may use a different reward_class than the checkpoint).
# Usage: ./scripts/launch_vcl_resume.sh <config> <resume_checkpoint>
set -e

CONFIG=${1:?Usage: $0 <config> <resume_checkpoint>}
RESUME=${2:?Usage: $0 <config> <resume_checkpoint>}
RUN_NAME=$(grep '^run_name:' "$CONFIG" | awk '{print $2}')

if [ -z "$RUN_NAME" ]; then
    echo "Could not parse run_name from $CONFIG"
    exit 1
fi
if [ ! -f "$RESUME" ]; then
    echo "Resume checkpoint not found: $RESUME"
    exit 1
fi

source "$HOME/.local/bin/env"
mkdir -p "runs/$RUN_NAME"

tmux new-session -d -s train \
    "cd $(pwd) && source $HOME/.local/bin/env && \
     PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy uv run python -m pokerl.scripts.train --config $CONFIG --resume $RESUME \
     2>&1 | tee runs/$RUN_NAME/train.log"

echo "Launched warm-start run '$RUN_NAME' (resume from $RESUME) in tmux session 'train'."
echo "Log: tail -f runs/$RUN_NAME/train.log"
