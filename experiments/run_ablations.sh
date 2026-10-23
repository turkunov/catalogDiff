#!/bin/bash
# Run ablation variants in the background inside pl-app, one process per variant; logs next to the results.
# usage (on the server, from the app directory): experiments/run_ablations.sh base photo_only ...
set -euo pipefail
LOGS=/data/pseudolabel/ablations/2026-10-23/logs
docker exec pl-app mkdir -p "$LOGS"
for v in "$@"; do
  docker exec -d pl-app sh -c "cd /workspace/app && python -m pseudolabel ablate --config experiments/ablations.yaml --variant $v > $LOGS/$v.log 2>&1"
  echo "started $v"
done
