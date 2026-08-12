#!/usr/bin/env bash
set -euo pipefail

# FastWAM RoboTwin training launcher (使用 scripts/train_zero1.sh).
#
# Default:
#   bash train_robotwin.sh
#
# Common overrides:
#   NPROC_PER_NODE=4 bash train_robotwin.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash train_robotwin.sh
#   RUN_ID=my_joint_run bash train_robotwin.sh
#   DRY_RUN=1 bash train_robotwin.sh
#
# Additional arguments are forwarded as Hydra overrides:
#   bash train_robotwin.sh batch_size=8 gradient_accumulation_steps=2
#
# ============================================================================
# 抢占恢复 (Auto-Resume):
#   - 在下面 "USER CONFIG" 区把 RUN_ID 设置为一个固定字符串 (如 my_joint_run)。
#   - 首次启动: 正常从头训练,checkpoint 会存到 runs/<task>/<RUN_ID>/checkpoints/。
#   - 机器被抢占重启后,只要以同样的 RUN_ID 再次调用本脚本 (`bash train_robotwin.sh`),
#     脚本会自动从 runs/<task>/<RUN_ID>/checkpoints/state/ 下的最新
#     step_XXXXXX 恢复 (optimizer/scheduler/step/dataloader 全部恢复)。
#   - 如果 checkpoints/state/ 不存在或为空,则从头训练 (不影响首次启动).
#   - 手工指定 resume=<路径> 时会覆盖自动检测。
# ============================================================================

# =============== USER CONFIG ================================================
# 指定 RUN_ID。留空 (:=) 才走时间戳默认值;设置成固定字符串以便 resume。
# 例:  RUN_ID_DEFAULT="robotwin_joint_3cam_384_1e-4_run01"
RUN_ID_DEFAULT=""
# ============================================================================

FASTWAM_ROOT="/ytech_milm_intern/zengxin08/FastWAM"
CONDA_ROOT="/ytech_milm_intern/zengxin08/miniconda3"
CONDA_ENV="/ytech_milm_intern/zengxin08/miniconda3/envs/fastwam"

: "${NPROC_PER_NODE:=8}"
: "${TASK_NAME:=robotwin_joint_3cam_384_1e-4}"
if [[ -n "${RUN_ID_DEFAULT}" ]]; then
    : "${RUN_ID:=${RUN_ID_DEFAULT}}"
else
    : "${RUN_ID:=$(TZ=CST-8 date '+%Y-%m-%d_%H-%M')}"
fi
: "${DRY_RUN:=0}"

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
    exit 1
fi

if [[ ! -d "${FASTWAM_ROOT}" ]]; then
    echo "ERROR: FastWAM root not found: ${FASTWAM_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda initialization script not found under: ${CONDA_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: FastWAM Python environment not found: ${CONDA_ENV}" >&2
    exit 1
fi

cd "${FASTWAM_ROOT}"

# Activate the exact environment instead of relying on the caller's shell state.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Put Python bytecode on local storage to reduce shared-filesystem metadata I/O.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_fastwam_$(id -u)}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TZ="${TZ:-Asia/Shanghai}"
# Persist HuggingFace datasets/hub cache on Ceph so Arrow fingerprints survive
# container restarts and can be shared across machines mounting /ytech_milm_intern.
export HF_HOME="${HF_HOME:-/ytech_milm_intern/zengxin08/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
# 利用碎片化显存
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RUN_ID

TASK_CONFIG="configs/task/${TASK_NAME}.yaml"
DATASET_ROOT="data/robotwin2.0/robotwin2.0"
DATASET_STATS="data/robotwin2.0/dataset_stats.json"
TEXT_CACHE="data/text_embeds_cache/robotwin"
ACTION_DIT_CKPT="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

for required_path in \
    "scripts/train_zero1.sh" \
    "${TASK_CONFIG}" \
    "${DATASET_ROOT}" \
    "${DATASET_STATS}" \
    "${TEXT_CACHE}" \
    "${ACTION_DIT_CKPT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required path not found: ${FASTWAM_ROOT}/${required_path}" >&2
        exit 1
    fi
done

if [[ -z "$(find -L "${TEXT_CACHE}" -maxdepth 1 -type f \
    -name '*.t5_len128.wan22ti2v5b.pt' -print -quit)" ]]; then
    echo "ERROR: no compatible T5 cache found under: ${FASTWAM_ROOT}/${TEXT_CACHE}" >&2
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    if (( ${#visible_gpus[@]} < NPROC_PER_NODE )); then
        echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE}, but CUDA_VISIBLE_DEVICES exposes only ${#visible_gpus[@]} GPUs." >&2
        exit 1
    fi
fi

OUTPUT_DIR="${FASTWAM_ROOT}/runs/${TASK_NAME}/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Auto-resume: 如果 OUTPUT_DIR 下已有完整的 state checkpoint,自动传 resume=<最新state>。
# 如果用户手工在命令行传了 resume=... 则跳过自动检测,尊重用户显式选择。
# ---------------------------------------------------------------------------
USER_RESUME_OVERRIDE=0
USER_RESUME_VALUE=""
for arg in "$@"; do
    if [[ "${arg}" == resume=* ]]; then
        USER_RESUME_OVERRIDE=1
        USER_RESUME_VALUE="${arg#resume=}"
        break
    fi
done

RESUME_ARGS=()
STATE_ROOT="${OUTPUT_DIR}/checkpoints/state"
if (( USER_RESUME_OVERRIDE == 0 )) && [[ -d "${STATE_ROOT}" ]]; then
    LATEST_STATE=$(ls -1 "${STATE_ROOT}" 2>/dev/null \
        | grep -E '^step_[0-9]+$' \
        | sort -V \
        | tail -n 1 || true)
    if [[ -n "${LATEST_STATE}" ]] && [[ -d "${STATE_ROOT}/${LATEST_STATE}" ]]; then
        RESUME_PATH="${STATE_ROOT}/${LATEST_STATE}"
        RESUME_ARGS=("resume=${RESUME_PATH}")
        echo "[auto-resume] Found existing checkpoint, will resume from: ${RESUME_PATH}"
    else
        echo "[auto-resume] ${STATE_ROOT} exists but no valid step_XXXXXX subdir; starting fresh."
    fi
elif (( USER_RESUME_OVERRIDE == 1 )); then
    echo "[auto-resume] Skipped (user provided explicit resume= override)."
fi

TRAIN_CMD=(
    bash scripts/train_zero1.sh
    "${NPROC_PER_NODE}"
    "task=${TASK_NAME}"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    "$@"
)

echo "========================================================================================================================"
echo "FastWAM RoboTwin training"
echo "  root              : ${FASTWAM_ROOT}"
echo "  python            : $(command -v python)"
echo "  task              : ${TASK_NAME}"
echo "  DeepSpeed         : ZeRO-1"
echo "  processes / GPUs  : ${NPROC_PER_NODE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "  run id            : ${RUN_ID}"
echo "  output            : ${OUTPUT_DIR}"
echo "  log               : ${LOG_FILE}"
if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "  resume            : ${RESUME_ARGS[0]#resume=} (auto-detected)"
elif (( USER_RESUME_OVERRIDE == 1 )); then
    echo "  resume            : ${USER_RESUME_VALUE} (user-specified)"
else
    echo "  resume            : <none> (fresh training)"
fi
printf '  command           :'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
echo "========================================================================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] Preflight checks passed; training was not started."
    exit 0
fi

"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
