#!/usr/bin/env bash
set -euo pipefail

# Version-2 multi-node launcher. Run the identical command on every node.
REPRESENTATION="${1:?Usage: $0 <dinov3_k7|vjepa2_1_vitg_causal_pair> <uncond|joint|idm> [Hydra overrides...]}"
MODE="${2:?Usage: $0 <dinov3_k7|vjepa2_1_vitg_causal_pair> <uncond|joint|idm> [Hydra overrides...]}"
shift 2

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FASTWAM_ROOT}"

case "${REPRESENTATION}" in
  dinov3_k7)
    DEFAULT_V2_STATS="checkpoints/DINOv3-L-K7-d1024/robotwin_clean2500_train_stats.pt"
    V1_TRANSPORT_REPRESENTATION="dinov3_k7"
    ;;
  vjepa2_1_vitg_causal_pair)
    DEFAULT_V2_STATS="checkpoints/VJEPA2.1-ViT-G-2B/robotwin_clean2500_train_stats_causal_pair.pt"
    V1_TRANSPORT_REPRESENTATION="dinov3_k7"
    ;;
  *) echo "ERROR: unsupported V2 representation: ${REPRESENTATION}; expected dinov3_k7 or vjepa2_1_vitg_causal_pair" >&2; exit 2 ;;
esac

: "${TRAIN_SEED:=42}"
: "${RAE_STATS_PATH:=${DEFAULT_V2_STATS}}"
: "${TASK_NAME:=robotwin_${MODE}_3cam_384_1e-4}"
export RAE_STATS_PATH
export TASK_NAME

for required in \
  "data/robotwin2.0/dataset_stats_clean2500_train.json" \
  "${RAE_STATS_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: version-2 stats file not found: ${FASTWAM_ROOT}/${required}" >&2
    exit 1
  fi
done

V2_TASK_NAME="robotwin_${MODE}_3cam_384_rae_${REPRESENTATION}_1e-4"
if [[ ! -f "configs/task/${V2_TASK_NAME}.yaml" ]]; then
  echo "ERROR: V2 task config not found: ${FASTWAM_ROOT}/configs/task/${V2_TASK_NAME}.yaml" >&2
  exit 1
fi
if [[ "${REPRESENTATION}" == vjepa2_1_* ]]; then
  : "${VJEPA2_REPO_PATH:=${FASTWAM_ROOT}/third_party/vjepa2}"
  export VJEPA2_REPO_PATH
  for required in \
    "checkpoints/VJEPA2.1-ViT-G-2B/vjepa2_1_vitG_384.pt" \
    "${VJEPA2_REPO_PATH}/app/vjepa_2_1/models/vision_transformer.py"; do
    if [[ ! -f "${required}" ]]; then
      echo "ERROR: required V-JEPA V2 path not found: ${required}" >&2
      exit 1
    fi
  done
fi

# Reuse the proven V1 multi-node topology/LR/resume implementation unchanged.
# Its positional representation selects only launcher-side defaults; this final
# Hydra task override selects the real V2 DINO/V-JEPA model configuration.
echo "V2 representation: ${REPRESENTATION}"
echo "V2 task config: ${V2_TASK_NAME}"
echo "V2 stats: ${RAE_STATS_PATH}"
exec bash "${FASTWAM_ROOT}/exp_v1/train_robotwin_mnode_rae.sh" "${V1_TRANSPORT_REPRESENTATION}" "${MODE}" \
  +experiment_version=v2_clean2500_randominit \
  "seed=${TRAIN_SEED}" \
  "task=${V2_TASK_NAME}" \
  "wandb.name=${V2_TASK_NAME}" \
  "$@"
