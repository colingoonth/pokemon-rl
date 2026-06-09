#!/usr/bin/env bash
# Run once on a fresh VCL reservation to get the environment ready.
set -e

# System deps
sudo apt-get install -y tmux

# uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env"

# Python deps
uv sync

echo "Setup complete. Run: ./scripts/launch_vcl.sh <config>"
