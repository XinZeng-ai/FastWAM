#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FASTWAM_ROOT}"

CONDA_ROOT="/ytech_milm_intern/zengxin08/miniconda3"
CONDA_ENV="${CONDA_ROOT}/envs/fastwam"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

: "${PAIR_GPU:=0}"
: "${VJEPA_STATS_BATCH:=256}"
: "${NUM_WORKERS:=4}"
: "${VJEPA2_REPO_PATH:=${FASTWAM_ROOT}/third_party/vjepa2}"
export VJEPA2_REPO_PATH
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

LOG_DIR="${FASTWAM_ROOT}/runs/rae_stats_clean2500/vjepa2_1"
mkdir -p "${LOG_DIR}"

echo "Computing V-JEPA causal-pair stats on GPU ${PAIR_GPU}"
echo "Log: ${LOG_DIR}/causal_pair.log"
CUDA_VISIBLE_DEVICES="${PAIR_GPU}" python scripts/compute_robotwin_rae_stats.py \
  --representation vjepa2_1_vitg_causal_pair \
  --output checkpoints/VJEPA2.1-ViT-G-2B/robotwin_clean2500_train_stats_causal_pair.pt \
  --data-config robotwin_clean2500 \
  --unique-frames \
  --episode-streaming \
  --batch-size "${VJEPA_STATS_BATCH}" \
  --num-workers "${NUM_WORKERS}" \
  --encoder-dtype bfloat16 \
  >"${LOG_DIR}/causal_pair.log" 2>&1
echo "V-JEPA causal-pair stats completed successfully."
