#!/usr/bin/env bash
set -euo pipefail

# Version-2 single-node launcher: clean 2500 episodes + random Video DiT +
# Action DiT initialized from that random Video DiT by interpolation/alpha scale.
REPRESENTATION="${1:?Usage: $0 <dinov3_k7> <uncond|joint|idm> [Hydra overrides...]}"
MODE="${2:?Usage: $0 <dinov3_k7> <uncond|joint|idm> [Hydra overrides...]}"
shift 2

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FASTWAM_ROOT}"

case "${REPRESENTATION}" in
  dinov3_k7) DEFAULT_V2_STATS="checkpoints/DINOv3-L-K7-d1024/robotwin_clean2500_train_stats.pt" ;;
  *) echo "ERROR: unsupported V2 representation: ${REPRESENTATION}; expected dinov3_k7" >&2; exit 2 ;;
esac

: "${TRAIN_SEED:=42}"
: "${RAE_STATS_PATH:=${DEFAULT_V2_STATS}}"
export RAE_STATS_PATH

for required in \
  "data/robotwin2.0/dataset_stats_clean2500_train.json" \
  "${RAE_STATS_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: version-2 stats file not found: ${FASTWAM_ROOT}/${required}" >&2
    exit 1
  fi
done

exec bash "${FASTWAM_ROOT}/exp_v1/train_robotwin_rae.sh" "${REPRESENTATION}" "${MODE}" \
  +experiment_version=v2_clean2500_randominit \
  "seed=${TRAIN_SEED}" \
  "$@"
