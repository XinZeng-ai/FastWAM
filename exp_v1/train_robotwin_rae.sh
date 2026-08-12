#!/usr/bin/env bash
set -euo pipefail

# Launch one of the 3 representations x 3 FastWAM modes.
#
#   DRY_RUN=1 bash train_robotwin_rae.sh siglip2_b joint
#
# RAE_STATS_PATH is selected automatically from the representation's
# robotwin_train_stats.pt. It can still be overridden explicitly when
# validating a different stats file.
# TASK_NAME controls the output directory in the same way as
# train_robotwin_mnode.sh. The internal RAE task config is selected from the
# representation/mode arguments and is independent of TASK_NAME.
#
# Extra arguments are forwarded as Hydra overrides.

REPRESENTATION="${1:?Usage: $0 <dinov3_k7|siglip2_b|mae_b> <uncond|joint|idm> [Hydra overrides...]}"
MODE="${2:?Usage: $0 <dinov3_k7|siglip2_b|mae_b> <uncond|joint|idm> [Hydra overrides...]}"
shift 2

# Resolve FASTWAM_ROOT from the script's location and cd there immediately so
# that all relative paths below work regardless of the caller's CWD.
FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FASTWAM_ROOT}"

case "${REPRESENTATION}" in
  dinov3_k7)
    DEFAULT_STATS="checkpoints/DINOv3-L-K7-d1024/robotwin_train_stats.pt"
    REQUIRED_ENCODER="checkpoints/DINOv3-L-K7-d1024/encoder/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    REQUIRED_DECODER="checkpoints/DINOv3-L-K7-d1024/decoder/decoder.pt"
    : "${DINOV3_REPO_PATH:=${FASTWAM_ROOT}/third_party/dinov3}"
    if [[ ! -f "${DINOV3_REPO_PATH}/hubconf.py" ]]; then
      echo "ERROR: official DINOv3 source checkout not found at ${DINOV3_REPO_PATH}." >&2
      echo "Set DINOV3_REPO_PATH or run: git clone --depth 1 https://github.com/facebookresearch/dinov3.git third_party/dinov3" >&2
      exit 1
    fi
    export DINOV3_REPO_PATH
    ;;
  siglip2_b)
    DEFAULT_STATS="checkpoints/SigLIP2-B-d768/robotwin_train_stats.pt"
    REQUIRED_ENCODER="checkpoints/SigLIP2-B-d768/encoder/model.safetensors"
    REQUIRED_DECODER="checkpoints/SigLIP2-B-d768/decoder/model.pt"
    ;;
  mae_b)
    DEFAULT_STATS="checkpoints/MAE-B-d768/robotwin_train_stats.pt"
    REQUIRED_ENCODER="checkpoints/MAE-B-d768/facebook/vit-mae-base/model.safetensors"
    REQUIRED_DECODER="checkpoints/MAE-B-d768/decoder/model.pt"
    ;;
  *)
    echo "ERROR: unknown representation '${REPRESENTATION}'; expected dinov3_k7, siglip2_b, or mae_b." >&2
    exit 2
    ;;
esac

case "${MODE}" in
  uncond|joint|idm) ;;
  *)
    echo "ERROR: unknown mode '${MODE}'; expected uncond, joint, or idm." >&2
    exit 2
    ;;
esac

: "${NPROC_PER_NODE:=8}"
: "${DRY_RUN:=0}"
# Use the project's pinned environment by default. PYTHON_BIN remains an
# override for another compatible environment or a dry-run-only setup.
: "${PYTHON_BIN:=/ytech_milm_intern/zengxin08/miniconda3/envs/fastwam/bin/python}"
: "${RAE_STATS_PATH:=${DEFAULT_STATS}}"
export RAE_STATS_PATH

: "${TASK_NAME:=robotwin_${MODE}_3cam_384_rae_${REPRESENTATION}_1e-4}"
: "${RUN_ID:=$(TZ=CST-8 date '+%Y-%m-%d_%H-%M')}"
RAE_TASK_NAME="robotwin_${MODE}_3cam_384_rae_${REPRESENTATION}_1e-4"
for required in \
  "configs/task/${RAE_TASK_NAME}.yaml" \
  "${REQUIRED_ENCODER}" \
  "${REQUIRED_DECODER}" \
  "${RAE_STATS_PATH}" \
  "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required RAE/FastWAM path not found: ${FASTWAM_ROOT}/${required}" >&2
    exit 1
  fi
done

TRAIN_CMD=(
  bash scripts/train_zero1.sh
  "${NPROC_PER_NODE}"
  "task=${RAE_TASK_NAME}"
  "output_dir=./runs/${TASK_NAME}/${RUN_ID}"
  "$@"
)

export RUN_ID
printf 'RAE config task: %s\n' "${RAE_TASK_NAME}"
printf 'Output task: %s\n' "${TASK_NAME}"
printf 'Stats: %s\n' "${RAE_STATS_PATH}"
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

# Activate the pinned conda environment so that `accelerate` and all
# dependencies (hydra, fastwam, torch …) resolve correctly.
CONDA_ROOT="/ytech_milm_intern/zengxin08/miniconda3"
CONDA_ENV="/ytech_milm_intern/zengxin08/miniconda3/envs/fastwam"
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda not found at ${CONDA_ROOT}" >&2
  exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_fastwam_$(id -u)}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TZ="${TZ:-Asia/Shanghai}"
export HF_HOME="${HF_HOME:-/ytech_milm_intern/zengxin08/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${DRY_RUN}" == "1" ]]; then
  if ! "${PYTHON_BIN}" -c "import hydra, fastwam" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN=${PYTHON_BIN} is not a FastWAM environment (hydra/fastwam import failed)." >&2
    echo "Activate the FastWAM environment or set PYTHON_BIN=/path/to/its/python." >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/train.py --cfg job "task=${RAE_TASK_NAME}" "$@" >/dev/null
  echo "[dry-run] Hydra composition and all local path checks passed; training was not started."
  exit 0
fi

OUTPUT_DIR="${FASTWAM_ROOT}/runs/${TASK_NAME}/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"
mkdir -p "${OUTPUT_DIR}"

"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
