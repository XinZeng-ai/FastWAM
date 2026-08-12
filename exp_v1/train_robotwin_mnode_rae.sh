#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# FastWAM RoboTwin RAE 多机训练启动器 (N × 8 GPU)
#
# 用法 (所有节点跑完全相同的命令):
#   NPROC_PER_NODE=8 EXPECTED_NNODES=2 \
#   bash train_robotwin_mnode_rae.sh dinov3_k7 uncond batch_size=1
#
# 位置参数:
#   $1 = representation: dinov3_k7 | siglip2_b | mae_b
#   $2 = mode:           uncond | joint | idm
#   剩余参数作为 Hydra overrides 转发
#
# 关键行为:
#   * 自动从 /etc/kml/ssh_configmap/pod_list 或 /etc/mpi/hostfile 解析
#     MASTER_ADDR / NODE_RANK; MY_NODE_IP 环境变量必须由平台注入。
#   * 期望 EXPECTED_NNODES 个节点, 等待 hostfile 写满再启动。
#   * 加载 KCCL (LD_PRELOAD) 加速跨节点通信, fallback 到原生 NCCL。
#   * 自动 resume: 相同 RUN_ID 再次启动会从 checkpoints/state/ 恢复。
#   * lr 按 global batch 相对基线做 sqrt 缩放。
#   * rank 0 写 train.log, 其他节点写 train.node<rank>.log
# ============================================================================

REPRESENTATION="${1:?Usage: $0 <dinov3_k7|siglip2_b|mae_b> <uncond|joint|idm> [Hydra overrides...]}"
MODE="${2:?Usage: $0 <dinov3_k7|siglip2_b|mae_b> <uncond|joint|idm> [Hydra overrides...]}"
shift 2

# ----------------------------------------------------------------------------
# FASTWAM_ROOT 从脚本位置解析, 支持在任意目录下运行
# ----------------------------------------------------------------------------
FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FASTWAM_ROOT}"

CONDA_ROOT="/ytech_milm_intern/zengxin08/miniconda3"
CONDA_ENV="/ytech_milm_intern/zengxin08/miniconda3/envs/fastwam"

# ----------------------------------------------------------------------------
# RAE 表示相关路径校验
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 环境变量默认值
# ----------------------------------------------------------------------------
: "${NPROC_PER_NODE:=8}"
: "${EXPECTED_NNODES:=2}"
: "${HOSTFILE_WAIT_SEC:=600}"
: "${DRY_RUN:=0}"
: "${MNODE_EPOCHS:=5}"
: "${MNODE_PER_RANK_BATCH:=1}"
: "${MNODE_SAVE_EVERY:=500}"
: "${MNODE_EVAL_EVERY:=200}"
: "${PYTHON_BIN:=${CONDA_ENV}/bin/python}"
: "${RAE_STATS_PATH:=${DEFAULT_STATS}}"
export RAE_STATS_PATH

: "${TASK_NAME:=robotwin_${MODE}_3cam_384_1e-4}"
: "${RUN_ID:=$(TZ=CST-8 date '+%Y-%m-%d_%H-%M')}"
RAE_TASK_NAME="robotwin_${MODE}_3cam_384_rae_${REPRESENTATION}_1e-4"

# ----------------------------------------------------------------------------
# 基础校验
# ----------------------------------------------------------------------------
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found at ${CONDA_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: fastwam env not found: ${CONDA_ENV}" >&2
    exit 1
fi

for required in \
  "configs/task/${RAE_TASK_NAME}.yaml" \
  "${REQUIRED_ENCODER}" \
  "${REQUIRED_DECODER}" \
  "${RAE_STATS_PATH}" \
  "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  "scripts/train_zero1.sh" \
  "data/robotwin2.0/robotwin2.0" \
  "data/robotwin2.0/dataset_stats.json" \
  "data/text_embeds_cache/robotwin" ; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required path not found: ${FASTWAM_ROOT}/${required}" >&2
    exit 1
  fi
done

# ----------------------------------------------------------------------------
# 激活 conda 环境
# ----------------------------------------------------------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ----------------------------------------------------------------------------
# Python 缓存与运行时环境
# ----------------------------------------------------------------------------
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_fastwam_$(id -u)}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TZ="${TZ:-Asia/Shanghai}"
export HF_HOME="${HF_HOME:-/ytech_milm_intern/zengxin08/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ----------------------------------------------------------------------------
# Output directory
# ----------------------------------------------------------------------------
OUTPUT_DIR="${FASTWAM_ROOT}/runs/${TASK_NAME}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------------
# 解析多机拓扑: 优先 pod_list(kml), fallback hostfile(mpi)
# ----------------------------------------------------------------------------
_SHARED_HOSTFILE="${OUTPUT_DIR}/.hostfile_shared"
_SHARED_READY_MARKER="${_SHARED_HOSTFILE}.ready.${RUN_ID}"

if [[ -f /etc/kml/ssh_configmap/pod_list ]]; then
    _HOSTFILE=/etc/kml/ssh_configmap/pod_list
    _IP_COL=3
elif [[ -f /etc/mpi/hostfile ]]; then
    _HOSTFILE=/etc/mpi/hostfile
    _IP_COL=1
    rm -f "${_SHARED_HOSTFILE}".ready.* 2>/dev/null || true
    _TMP="${_SHARED_HOSTFILE}.tmp.$$"
    if cp /etc/mpi/hostfile "${_TMP}" && mv -f "${_TMP}" "${_SHARED_HOSTFILE}"; then
        : > "${_SHARED_READY_MARKER}"
        echo ">>> Synced /etc/mpi/hostfile -> ${_SHARED_HOSTFILE}"
    else
        echo ">>> WARNING: failed to sync hostfile to ${_SHARED_HOSTFILE}" >&2
        rm -f "${_TMP}" 2>/dev/null || true
    fi
else
    echo ">>> Waiting for shared hostfile marker: $(basename "${_SHARED_READY_MARKER}") ..."
    _waited=0
    while [[ ! -f "${_SHARED_READY_MARKER}" ]] || [[ ! -f "${_SHARED_HOSTFILE}" ]]; do
        if (( _waited >= HOSTFILE_WAIT_SEC )); then
            echo "ERROR: timeout ${HOSTFILE_WAIT_SEC}s waiting for ${_SHARED_READY_MARKER}." >&2
            exit 1
        fi
        sleep 5
        _waited=$((_waited + 5))
        (( _waited % 30 == 0 )) && echo "    still waiting... (${_waited}s)"
    done
    _HOSTFILE="${_SHARED_HOSTFILE}"
    _IP_COL=1
    echo ">>> Using shared hostfile: ${_HOSTFILE}"
fi

echo ">>> Waiting for $_HOSTFILE to contain ${EXPECTED_NNODES} entries..."
_waited=0
while :; do
    _cur=$(awk 'END{print NR}' "$_HOSTFILE")
    if [[ "${_cur}" -ge "${EXPECTED_NNODES}" ]]; then
        echo ">>> hostfile ready: ${_cur}/${EXPECTED_NNODES} lines."
        break
    fi
    if [[ ${_waited} -ge ${HOSTFILE_WAIT_SEC} ]]; then
        echo "ERROR: hostfile only has ${_cur}/${EXPECTED_NNODES} entries after ${HOSTFILE_WAIT_SEC}s." >&2
        exit 1
    fi
    echo "    hostfile has ${_cur}/${EXPECTED_NNODES} lines, sleeping 5s... (${_waited}s)"
    sleep 5
    _waited=$((_waited + 5))
done

export MASTER_ADDR=$(awk -v c="$_IP_COL" 'NR==1{print $c; exit}' "$_HOSTFILE")
export NNODES="${EXPECTED_NNODES}"
export NODE_RANK=$(awk -v ip="${MY_NODE_IP:-}" -v c="$_IP_COL" -v n="$EXPECTED_NNODES" \
    'NR<=n && $c==ip{print NR-1; exit}' "$_HOSTFILE")

if [[ -z "${NODE_RANK}" ]]; then
    echo "ERROR: Cannot resolve NODE_RANK. MY_NODE_IP=${MY_NODE_IP:-<unset>} not in first ${EXPECTED_NNODES} lines of $_HOSTFILE." >&2
    exit 1
fi

: "${MASTER_PORT:=29500}"
TOTAL_GPU=$(( NNODES * NPROC_PER_NODE ))
DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-/tmp/fastwam_deepspeed_${RUN_ID}.hostfile}"
mapfile -t _DEEPSPEED_HOSTS < <(awk -v c="${_IP_COL}" -v n="${EXPECTED_NNODES}" '
    NR <= n {
        if ($c == "" || seen[$c]++) exit 1
        print $c
    }
    END { if (NR < n) exit 1 }
' "${_HOSTFILE}")
if [[ "${#_DEEPSPEED_HOSTS[@]}" -ne "${EXPECTED_NNODES}" ]]; then
    echo "ERROR: Cannot create a valid DeepSpeed hostfile from ${_HOSTFILE}." >&2
    exit 1
fi
_DEEPSPEED_HOSTFILE_TMP="${DEEPSPEED_HOSTFILE}.tmp.$$"
{
    for _host in "${_DEEPSPEED_HOSTS[@]}"; do
        printf '%s slots=%s\n' "${_host}" "${NPROC_PER_NODE}"
    done
} > "${_DEEPSPEED_HOSTFILE_TMP}"
mv -f "${_DEEPSPEED_HOSTFILE_TMP}" "${DEEPSPEED_HOSTFILE}"

# Extract gradient_accumulation_steps from Hydra overrides (if present)
MNODE_GRAD_ACCUM=1
for _arg in "$@"; do
    if [[ "${_arg}" == gradient_accumulation_steps=* ]]; then
        MNODE_GRAD_ACCUM="${_arg#gradient_accumulation_steps=}"
        break
    fi
done

GLOBAL_BATCH=$(( TOTAL_GPU * MNODE_PER_RANK_BATCH * MNODE_GRAD_ACCUM ))

# LR sqrt scaling
: "${MNODE_LR_REF:=1e-4}"
: "${MNODE_LR_REF_BATCH:=1024}"
LR=$(awk -v b="${MNODE_LR_REF}" -v gb="${GLOBAL_BATCH}" -v ref="${MNODE_LR_REF_BATCH}" \
    'BEGIN{printf "%.4e", b * sqrt(gb / ref)}')

# ----------------------------------------------------------------------------
# NCCL / KCCL
# ----------------------------------------------------------------------------
_CUDA13_LIB="/usr/local/cuda-13.2/targets/x86_64-linux/lib"
: "${KCCL_LIB:=/opt/kccl/ubuntu/cuda13/libkccl.so.2.1-2.29.7.ubuntu-cuda13}"
: "${USE_KCCL:=0}"

_TORCH_CUDA_MAJOR=$("${CONDA_ENV}/bin/python" -c "import torch,sys; sys.stdout.write((torch.version.cuda or '0').split('.')[0])" 2>/dev/null || echo "?")

if [[ "${USE_KCCL:-auto}" == "0" ]]; then
    echo ">>> KCCL disabled by USE_KCCL=0. Using native NCCL."
elif [[ "${_TORCH_CUDA_MAJOR}" != "13" ]]; then
    echo ">>> KCCL skipped: PyTorch CUDA=${_TORCH_CUDA_MAJOR} incompatible with KCCL (cuda13 only)."
elif [[ -f "${KCCL_LIB}" ]] && [[ -d "${_CUDA13_LIB}" ]]; then
    export LD_PRELOAD="${KCCL_LIB}:${_CUDA13_LIB}/libcublasLt.so.13:${_CUDA13_LIB}/libcublas.so.13${LD_PRELOAD:+:${LD_PRELOAD}}"
    export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
    echo ">>> KCCL loaded: ${KCCL_LIB}"
else
    echo ">>> WARNING: KCCL not found, falling back to native NCCL." >&2
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
: "${FORCE_IB:=1}"
if [[ "${FORCE_IB}" == "1" ]]; then
    export NCCL_IB_DISABLE=0
    echo ">>> FORCE_IB=1: overriding NCCL_IB_DISABLE -> 0"
fi
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"

if [[ -z "${NCCL_IB_HCA:-}" ]]; then
    _BEST_HCA=""
    _BEST_MTU=0
    if command -v ibdev2netdev &>/dev/null && command -v ibv_devinfo &>/dev/null; then
    while read -r _line; do
        [[ "${_line}" == *"(Up)" ]] || continue
        _ibdev=$(echo "${_line}" | awk '{print $1}')
        _mtu=$(ibv_devinfo -d "${_ibdev}" 2>/dev/null \
            | awk -F'[[:space:]]+' '/active_mtu/{print $3; exit}' 2>/dev/null || echo 0)
        _mtu=${_mtu:-0}
        if (( _mtu > _BEST_MTU )); then
            _BEST_MTU=${_mtu}
            _BEST_HCA="${_ibdev}"
        fi
    done < <(ibdev2netdev 2>/dev/null)
    fi
    if [[ -n "${_BEST_HCA}" ]]; then
        export NCCL_IB_HCA="=${_BEST_HCA}"
        echo ">>> Auto-detected NCCL_IB_HCA=${_BEST_HCA} (active_mtu=${_BEST_MTU})"
    else
        echo ">>> WARNING: Could not auto-detect best IB HCA; leaving unset."
    fi
    unset _BEST_HCA _BEST_MTU _ibdev _mtu _line
else
    export NCCL_IB_HCA
fi

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    _ROUTE_PEER=""
    for _host in "${_DEEPSPEED_HOSTS[@]}"; do
        if [[ "${_host}" != "${MY_NODE_IP:-}" ]]; then
            _ROUTE_PEER="${_host}"
            break
        fi
    done
    if [[ -z "${_ROUTE_PEER}" ]]; then
        echo "ERROR: Cannot determine an NCCL route peer from the DeepSpeed hostfile." >&2
        exit 1
    fi
    NCCL_SOCKET_IFNAME="$(ip route get "${_ROUTE_PEER}" 2>/dev/null | awk '{for (i=1;i<=NF;++i) if ($i=="dev") {print $(i+1); exit}}')"
    if [[ -z "${NCCL_SOCKET_IFNAME}" ]]; then
        echo "ERROR: Cannot resolve NIC route to ${_ROUTE_PEER}; set NCCL_SOCKET_IFNAME explicitly." >&2
        exit 1
    fi
fi
export NCCL_SOCKET_IFNAME
echo ">>> NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_ECE_ENABLE="${NCCL_IB_ECE_ENABLE:-0}"
export NCCL_IB_ADAPTIVE_ROUTING="${NCCL_IB_ADAPTIVE_ROUTING:-0}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-13}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring}"
export NCCL_PROTO="${NCCL_PROTO:-Simple,LL,LL128}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-3600}"

# ----------------------------------------------------------------------------
# 清理 MASTER_PORT 残留 (rank 0 才做)
# ----------------------------------------------------------------------------
if [[ "${NODE_RANK}" -eq 0 ]]; then
    echo ">>> Cleaning stale processes on port ${MASTER_PORT}..."
    pkill -f "accelerate.*${MASTER_PORT}" 2>/dev/null || true
    sleep 2
    _wait=0
    while ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT}\\b"; do
        [[ ${_wait} -ge 120 ]] && break
        echo "    port ${MASTER_PORT} still in use, waiting... (${_wait}s)"
        sleep 5
        _wait=$((_wait + 5))
    done
fi

# ----------------------------------------------------------------------------
# Auto-resume
# ----------------------------------------------------------------------------
if [[ "${NODE_RANK}" -eq 0 ]]; then
    LOG_FILE="${OUTPUT_DIR}/train.log"
else
    LOG_FILE="${OUTPUT_DIR}/train.node${NODE_RANK}.log"
fi

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
        RESUME_ARGS=("resume=${STATE_ROOT}/${LATEST_STATE}")
        echo "[auto-resume] Found existing checkpoint, will resume from: ${STATE_ROOT}/${LATEST_STATE}"
    fi
elif (( USER_RESUME_OVERRIDE == 1 )); then
    echo "[auto-resume] Skipped (user explicit resume=${USER_RESUME_VALUE})"
fi

export RUN_ID

# ----------------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
    if ! "${PYTHON_BIN}" -c "import hydra, fastwam" >/dev/null 2>&1; then
        echo "ERROR: PYTHON_BIN=${PYTHON_BIN} is not a FastWAM environment." >&2
        exit 1
    fi
    "${PYTHON_BIN}" scripts/train.py --cfg job "task=${RAE_TASK_NAME}" "$@" >/dev/null
    echo "[dry-run] Hydra composition and all local path checks passed; training was not started."
    exit 0
fi

# ----------------------------------------------------------------------------
# 打印摘要
# ----------------------------------------------------------------------------
echo "========================================================================================================================"
echo "FastWAM RoboTwin RAE MULTI-NODE training"
echo "  root              : ${FASTWAM_ROOT}"
echo "  representation    : ${REPRESENTATION}"
echo "  mode              : ${MODE}"
echo "  hostfile          : ${_HOSTFILE}"
echo "  MY_NODE_IP        : ${MY_NODE_IP:-<unset>}"
echo "  MASTER_ADDR       : ${MASTER_ADDR}"
echo "  MASTER_PORT       : ${MASTER_PORT}"
echo "  NNODES            : ${NNODES}"
echo "  NODE_RANK         : ${NODE_RANK}"
echo "  NPROC_PER_NODE    : ${NPROC_PER_NODE}"
echo "  TOTAL_GPU         : ${TOTAL_GPU}"
echo "  RUN_ID            : ${RUN_ID}"
echo "  RAE task          : ${RAE_TASK_NAME}"
echo "  output task       : ${TASK_NAME}"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "  LOG_FILE          : ${LOG_FILE}"
echo "  epochs            : ${MNODE_EPOCHS}"
echo "  per-rank batch    : ${MNODE_PER_RANK_BATCH}"
echo "  grad_accum        : ${MNODE_GRAD_ACCUM}"
echo "  GLOBAL_BATCH      : ${GLOBAL_BATCH}"
echo "  learning_rate     : ${LR}"
echo "  save_every        : ${MNODE_SAVE_EVERY}"
echo "  eval_every        : ${MNODE_EVAL_EVERY}"
echo "  stats             : ${RAE_STATS_PATH} (metadata validated by tokenizer)"
echo "  deepspeed hostfile: ${DEEPSPEED_HOSTFILE}"
if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "  resume            : ${RESUME_ARGS[0]#resume=} (auto-detected)"
elif (( USER_RESUME_OVERRIDE == 1 )); then
    echo "  resume            : ${USER_RESUME_VALUE} (user-specified)"
else
    echo "  resume            : <none> (fresh training)"
fi
echo "========================================================================================================================"

# ----------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------
exec accelerate launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_machines "${NNODES}" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --num_processes "${TOTAL_GPU}" \
    --deepspeed_hostfile "${DEEPSPEED_HOSTFILE}" \
    scripts/train.py \
    "output_dir=./runs/${TASK_NAME}/${RUN_ID}" \
    "wandb.name=${RAE_TASK_NAME}" \
    "task=${RAE_TASK_NAME}" \
    "batch_size=${MNODE_PER_RANK_BATCH}" \
    "num_epochs=${MNODE_EPOCHS}" \
    "learning_rate=${LR}" \
    "save_every=${MNODE_SAVE_EVERY}" \
    "eval_every=${MNODE_EVAL_EVERY}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
