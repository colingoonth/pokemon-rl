#!/usr/bin/env bash
# Launch a training run on VCL in a persistent tmux session.
# Usage: ./scripts/launch_vcl.sh <config>
# Example: ./scripts/launch_vcl.sh configs/vcl_brock_v0_4_2_center_1gpu.yaml
set -e

CONFIG=${1:?Usage: $0 <config>}
RUN_NAME=$(grep '^run_name:' "$CONFIG" | awk '{print $2}')

if [ -z "$RUN_NAME" ]; then
    echo "Could not parse run_name from $CONFIG"
    exit 1
fi

source "$HOME/.local/bin/env"
mkdir -p "runs/$RUN_NAME"

tmux new-session -d -s train \
    "cd $(pwd) && source $HOME/.local/bin/env && \
     PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy uv run python -m pokerl.scripts.train --config $CONFIG \
     2>&1 | tee runs/$RUN_NAME/train.log"

echo "Launched run '$RUN_NAME' in tmux session 'train'."
echo "Attach: tmux attach -t train"
echo "Log:    tail -f runs/$RUN_NAME/train.log"
