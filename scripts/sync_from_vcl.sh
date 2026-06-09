#!/usr/bin/env bash
# Pull run outputs from VCL to local.
# Usage: ./scripts/sync_from_vcl.sh <vcl_host> [run_name]
# Example: ./scripts/sync_from_vcl.sh cguenth2@152.7.176.245
#          ./scripts/sync_from_vcl.sh cguenth2@152.7.176.245 vcl_brock_v0_4_2_center_1gpu
set -e

VCL_HOST=${1:?Usage: $0 <vcl_host> [run_name]}
RUN_NAME=${2:-}

if [ -n "$RUN_NAME" ]; then
    rsync -avz "$VCL_HOST:~/pokemon-rl/runs/$RUN_NAME/" "runs/$RUN_NAME/"
else
    rsync -avz "$VCL_HOST:~/pokemon-rl/runs/" runs/
fi
