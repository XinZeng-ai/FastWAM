#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/ytech_milm_intern/zengxin08/FastWAM"
CONDA_BASE="/ytech_milm_intern/zengxin08/miniconda3"

# Make `conda activate fastwam` available in this non-interactive shell.
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate fastwam

cd "${PROJECT_ROOT}"

# Use China Standard Time (UTC+8) for Hydra output timestamps and Python logs.
# POSIX TZ signs are reversed, so CST-8 means UTC+8 and needs no zoneinfo file.
export TZ="CST-8"

# RoboTwin evaluates from third_party/RoboTwin; use an absolute base path so
# FastWAM can find the locally cached Wan VAE, T5 encoder, and tokenizer.
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_ROOT}/checkpoints"

# RoboTwin starts FFmpeg subprocesses after Hugging Face tokenization has run.
# Disable tokenizer CPU thread-pool parallelism to avoid fork-safety warnings.
export TOKENIZERS_PARALLELISM=false

# Restore NVIDIA's EGL/Vulkan discovery when the container's injected vendor
# manifest is missing or empty.
source "${PROJECT_ROOT}/fix_nvidia_rendering.sh"

# Usage:
#   bash eval_robotwin.sh [NUM_GPUS] [WORKERS_PER_GPU]
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 WORKERS_PER_GPU=2 \
#     TASK_CONFIG=robotwin_uncond_3cam_384_1e-4 \
#     EVAL_DATASET_STATS_PATH=./data/robotwin2.0/dataset_stats_clean2500_train.json \
#     EVAL_RAE_STATS_DATASET=robotwin_clean2500_train \
#     CKPT_PATH=/path/to/checkpoint.pt bash eval_robotwin.sh
#   TASK_CONFIG=robotwin_joint_3cam_384_1e-4 \
#     CKPT_PATH=/path/to/joint-checkpoint.pt bash eval_robotwin.sh
#   TASK_CONFIG=robotwin_idm_3cam_384_1e-4 \
#     CKPT_PATH=/path/to/idm-checkpoint.pt bash eval_robotwin.sh
#   # Resume a preempted run at phase granularity:
#   RESUME_EVAL=true EVAL_RUN_ID=20260803_103430 \
#     TASK_CONFIG=robotwin_joint_3cam_384_rae_mae_b_1e-4 \
#     CKPT_PATH=/path/to/joint-checkpoint.pt bash eval_robotwin.sh 8 2
NUM_GPUS="${NUM_GPUS:-${1:-2}}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-${2:-2}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MNODE_EVAL_SESSION_ID="${MNODE_EVAL_SESSION_ID:-${EVAL_RUN_ID:-single_node}}"
MNODE_COORD_TIMEOUT_SEC="${MNODE_COORD_TIMEOUT_SEC:-3600}"
TASK_CONFIG="${TASK_CONFIG:-robotwin_uncond_3cam_384_1e-4}"
CKPT_PATH="${CKPT_PATH:-./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
EVAL_DATASET_STATS_PATH="${EVAL_DATASET_STATS_PATH:-./data/robotwin2.0/dataset_stats.json}"
EVAL_RAE_STATS_DATASET="${EVAL_RAE_STATS_DATASET:-}"
EVAL_RAE_STATS_PATH="${EVAL_RAE_STATS_PATH:-}"
DEFAULT_SAVE_PREDICTED_VIDEO=false
case "${TASK_CONFIG}" in
  robotwin_joint_*|robotwin_idm_*) DEFAULT_SAVE_PREDICTED_VIDEO=true ;;
esac
SAVE_PREDICTED_VIDEO="${SAVE_PREDICTED_VIDEO:-${DEFAULT_SAVE_PREDICTED_VIDEO}}"
# 保存多少个 episode 预测视频
PREDICTED_VIDEO_MAX_EPISODES="${PREDICTED_VIDEO_MAX_EPISODES:-0}"
PREDICTED_VIDEO_MAX_REPLANS="${PREDICTED_VIDEO_MAX_REPLANS:-0}"
RESUME_EVAL="${RESUME_EVAL:-false}"
EVAL_RUN_ID="${EVAL_RUN_ID:-}"

normalize_bool() {
  local name="$1"
  local value="${2,,}"
  case "${value}" in
    1|true|yes|y) printf 'true' ;;
    0|false|no|n) printf 'false' ;;
    *)
      echo "${name} must be a boolean (true/false or 1/0), got: $2" >&2
      return 2
      ;;
  esac
}

SAVE_PREDICTED_VIDEO="$(normalize_bool SAVE_PREDICTED_VIDEO "${SAVE_PREDICTED_VIDEO}")"
RESUME_EVAL="$(normalize_bool RESUME_EVAL "${RESUME_EVAL}")"

if [[ "${RESUME_EVAL}" == "true" && -z "${EVAL_RUN_ID}" ]]; then
  echo "EVAL_RUN_ID is required when RESUME_EVAL=true." >&2
  exit 2
fi
if [[ -n "${EVAL_RUN_ID}" && ! "${EVAL_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EVAL_RUN_ID may contain only letters, digits, dot, underscore, and hyphen; got: ${EVAL_RUN_ID}" >&2
  exit 2
fi

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 2
fi
if ! [[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_GPU must be a positive integer, got: ${WORKERS_PER_GPU}" >&2
  exit 2
fi
if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer, got: ${NNODES}" >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ -z "${EVAL_RUN_ID}" ]]; then
  echo "EVAL_RUN_ID is required when NNODES>1 so every node shares one output directory." >&2
  exit 2
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, ${NNODES}), got: ${NODE_RANK}" >&2
  exit 2
fi
if ! [[ "${MNODE_COORD_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MNODE_COORD_TIMEOUT_SEC must be a positive integer, got: ${MNODE_COORD_TIMEOUT_SEC}" >&2
  exit 2
fi
if [[ ! "${MNODE_EVAL_SESSION_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "MNODE_EVAL_SESSION_ID may contain only letters, digits, dot, underscore, and hyphen; got: ${MNODE_EVAL_SESSION_ID}" >&2
  exit 2
fi
if ! [[ "${PREDICTED_VIDEO_MAX_EPISODES}" =~ ^[0-9]+$ ]]; then
  echo "PREDICTED_VIDEO_MAX_EPISODES must be a non-negative integer, got: ${PREDICTED_VIDEO_MAX_EPISODES}" >&2
  exit 2
fi
if ! [[ "${PREDICTED_VIDEO_MAX_REPLANS}" =~ ^[0-9]+$ ]]; then
  echo "PREDICTED_VIDEO_MAX_REPLANS must be a non-negative integer, got: ${PREDICTED_VIDEO_MAX_REPLANS}" >&2
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_DATASET_STATS_PATH}" ]]; then
  echo "Evaluation dataset stats not found: ${EVAL_DATASET_STATS_PATH}" >&2
  exit 2
fi
if [[ ! -f "configs/task/${TASK_CONFIG}.yaml" ]]; then
  echo "Task config not found: configs/task/${TASK_CONFIG}.yaml" >&2
  exit 2
fi
if [[ -n "${EVAL_RAE_STATS_DATASET}" ]]; then
  if [[ -z "${EVAL_RAE_STATS_PATH}" ]]; then
    case "${TASK_CONFIG}" in
      *rae_dinov3_k7*) rae_checkpoint_dir="checkpoints/DINOv3-L-K7-d1024" ;;
      *rae_siglip2_b*) rae_checkpoint_dir="checkpoints/SigLIP2-B-d768" ;;
      *rae_mae_b*) rae_checkpoint_dir="checkpoints/MAE-B-d768" ;;
      *)
        echo "EVAL_RAE_STATS_DATASET was set, but TASK_CONFIG does not identify a supported RAE: ${TASK_CONFIG}" >&2
        exit 2
        ;;
    esac
    EVAL_RAE_STATS_PATH="${rae_checkpoint_dir}/${EVAL_RAE_STATS_DATASET}_stats.pt"
  fi
  if [[ ! -f "${EVAL_RAE_STATS_PATH}" ]]; then
    echo "RAE evaluation stats not found: ${EVAL_RAE_STATS_PATH}" >&2
    exit 2
  fi
  # Make the requested evaluation dataset authoritative even if the caller's
  # shell still exports an RAE_STATS_PATH from an earlier experiment.
  export RAE_STATS_PATH="$(realpath "${EVAL_RAE_STATS_PATH}")"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible_devices=""
  for ((gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)); do
    if [[ -n "${visible_devices}" ]]; then
      visible_devices+=","
    fi
    visible_devices+="${gpu_idx}"
  done
  export CUDA_VISIBLE_DEVICES="${visible_devices}"
fi

OUTPUT_ARGS=()
if [[ -n "${EVAL_RUN_ID}" ]]; then
  OUTPUT_ARGS+=(
    "EVALUATION.output_dir=./evaluate_results/robotwin/${TASK_CONFIG}/${EVAL_RUN_ID}"
  )
fi
if [[ -n "${EVAL_RAE_STATS_DATASET}" ]]; then
  OUTPUT_ARGS+=(
    "data.rae_stats_dataset=${EVAL_RAE_STATS_DATASET}"
    "data.rae_stats_filename=${EVAL_RAE_STATS_DATASET}_stats.pt"
  )
fi

exec python experiments/robotwin/run_robotwin_manager.py \
  task="${TASK_CONFIG}" \
  ckpt="${CKPT_PATH}" \
  EVALUATION.dataset_stats_path="${EVAL_DATASET_STATS_PATH}" \
  EVALUATION.task_name=null \
  EVALUATION.resume_existing="${RESUME_EVAL}" \
  EVALUATION.save_predicted_video="${SAVE_PREDICTED_VIDEO}" \
  EVALUATION.predicted_video_max_episodes="${PREDICTED_VIDEO_MAX_EPISODES}" \
  EVALUATION.predicted_video_max_replans="${PREDICTED_VIDEO_MAX_REPLANS}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${WORKERS_PER_GPU}" \
  +MULTIRUN.num_nodes="${NNODES}" \
  +MULTIRUN.node_rank="${NODE_RANK}" \
  +MULTIRUN.session_id="${MNODE_EVAL_SESSION_ID}" \
  +MULTIRUN.coordination_timeout_sec="${MNODE_COORD_TIMEOUT_SEC}" \
  "${OUTPUT_ARGS[@]}"
