#!/usr/bin/env bash
set -euo pipefail

# Version-2 single-node VAE launcher. Select uncond/joint/idm through TASK_NAME
# exactly as in train_robotwin.sh.
FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FASTWAM_ROOT}"
: "${TRAIN_SEED:=42}"

if [[ ! -f data/robotwin2.0/dataset_stats_clean2500_train.json ]]; then
  echo "ERROR: version-2 dataset stats not found: ${FASTWAM_ROOT}/data/robotwin2.0/dataset_stats_clean2500_train.json" >&2
  exit 1
fi

exec bash "${FASTWAM_ROOT}/exp_v1/train_robotwin.sh" \
  +experiment_version=v2_clean2500_randominit \
  "seed=${TRAIN_SEED}" \
  "$@"
