#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# FastWAM RoboTwin 多机训练启动器 (N × b300/a800 × 8 GPU)
#
# 用法(所有节点跑完全相同的命令):
#   bash train_robotwin_mnode.sh
#
# 关键行为:
#   * EXPECTED_NNODES 固定期望节点数, 等待 hostfile 写满这么多行再启动,
#     避免 pod_list 分批到位导致不同节点看到不同拓扑、写坏 checkpoint。
#   * 自动从 /etc/kml/ssh_configmap/pod_list 或 /etc/mpi/hostfile 解析
#     MASTER_ADDR / NODE_RANK, MY_NODE_IP 环境变量必须由平台注入。
#   * 加载 KCCL (LD_PRELOAD) 加速跨节点通信,fallback 到原生 NCCL 时会打警告。
#   * 自动 resume: 相同 RUN_ID 再次启动会从 runs/<task>/<RUN_ID>/checkpoints/state/
#     的最新 step_XXXXXX 恢复。抢占重启场景直接生效。
#   * lr 按 global batch 相对基线做 sqrt 缩放, 基线 = 64卡 bs=16 = global 1024, lr=1e-4
#
# 常用环境变量覆盖:
#   EXPECTED_NNODES         期望节点数
#   HOSTFILE_WAIT_SEC=600     等 hostfile 写满的超时秒数
#   NPROC_PER_NODE=8          每节点 GPU 数
#   MNODE_EPOCHS=2            训练 epoch 数
#   MNODE_LR_REF=1e-4         基线 lr (对应 MNODE_LR_REF_BATCH 的 global batch)
#   MNODE_LR_REF_BATCH=1024    基线 global batch (原作者=64卡×bs16=1024)
#   MNODE_PER_RANK_BATCH   单 rank(单卡) batch size, a800用16, b300用64
#   MNODE_SAVE_EVERY     checkpoint 保存频率(steps)
#   MNODE_EVAL_EVERY      eval 频率(steps)
#   RUN_ID        固定 run id (便于 resume,留空则用时间戳)
#   TASK_NAME=robotwin_joint_3cam_384_1e-4
# ============================================================================

FASTWAM_ROOT="/ytech_milm_intern/zengxin08/FastWAM"
CONDA_ROOT="/ytech_milm_intern/zengxin08/miniconda3"
CONDA_ENV="/ytech_milm_intern/zengxin08/miniconda3/envs/fastwam"

: "${EXPECTED_NNODES:=2}"
: "${HOSTFILE_WAIT_SEC:=600}"
: "${NPROC_PER_NODE:=8}"
: "${TASK_NAME:=robotwin_joint_3cam_384_1e-4}"
: "${MNODE_EPOCHS:=2}"
: "${MNODE_PER_RANK_BATCH:=64}"
: "${MNODE_SAVE_EVERY:=500}"
: "${MNODE_EVAL_EVERY:=200}"
: "${DRY_RUN:=0}"
: "${RUN_ID:=$(TZ=CST-8 date '+%Y-%m-%d_%H-%M')}"

# ---------------------------------------------------------------------------
# 基础校验
# ---------------------------------------------------------------------------
if [[ ! -d "${FASTWAM_ROOT}" ]]; then
    echo "ERROR: FastWAM root not found: ${FASTWAM_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found: ${CONDA_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: fastwam env not found: ${CONDA_ENV}" >&2
    exit 1
fi

cd "${FASTWAM_ROOT}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Python 缓存与运行时环境
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------#
# Output directory (提前定义,供 shared hostfile 等临时文件存放)
# ---------------------------------------------------------------------------
OUTPUT_DIR="${FASTWAM_ROOT}/runs/${TASK_NAME}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 解析多机拓扑: 优先 pod_list(kml)、fallback hostfile(mpi)
# 等待 hostfile 写满 EXPECTED_NNODES 行再继续,避免节点分批上线导致拓扑不一致。
# ---------------------------------------------------------------------------
_SHARED_HOSTFILE="${OUTPUT_DIR}/.hostfile_shared"
_SHARED_READY_MARKER="${_SHARED_HOSTFILE}.ready.${RUN_ID}"

if [[ -f /etc/kml/ssh_configmap/pod_list ]]; then
    _HOSTFILE=/etc/kml/ssh_configmap/pod_list
    _IP_COL=3
elif [[ -f /etc/mpi/hostfile ]]; then
    _HOSTFILE=/etc/mpi/hostfile
    _IP_COL=1
    # 该节点有 /etc/mpi/hostfile (通常仅 node 0),同步到共享盘供其他节点 fallback。
    # 用 tmp+mv 保证原子替换,并写一个带 RUN_ID 的 ready marker 让子节点确认是本次运行的最新版本。
    rm -f "${_SHARED_HOSTFILE}".ready.* 2>/dev/null || true
    _TMP="${_SHARED_HOSTFILE}.tmp.$$"
    if cp /etc/mpi/hostfile "${_TMP}" && mv -f "${_TMP}" "${_SHARED_HOSTFILE}"; then
        : > "${_SHARED_READY_MARKER}"
        echo ">>> Synced /etc/mpi/hostfile -> ${_SHARED_HOSTFILE} (marker: $(basename "${_SHARED_READY_MARKER}"))"
    else
        echo ">>> WARNING: failed to sync hostfile to ${_SHARED_HOSTFILE}" >&2
        rm -f "${_TMP}" 2>/dev/null || true
    fi
else
    # 子节点:等待 node 0 生成本次 RUN_ID 对应的 ready marker (避免用到上次 run 的 stale 文件)
    echo ">>> $(TZ=Asia/Shanghai date '+%H:%M:%S') No local hostfile. Waiting for shared marker: $(basename "${_SHARED_READY_MARKER}") ..."
    _waited=0
    while [[ ! -f "${_SHARED_READY_MARKER}" ]] || [[ ! -f "${_SHARED_HOSTFILE}" ]]; do
        if (( _waited >= HOSTFILE_WAIT_SEC )); then
            echo "ERROR: timeout ${HOSTFILE_WAIT_SEC}s waiting for ${_SHARED_READY_MARKER}." >&2
            echo "       Ensure node 0 is running the same script with RUN_ID=${RUN_ID}." >&2
            exit 1
        fi
        sleep 5
        _waited=$((_waited + 5))
        (( _waited % 30 == 0 )) && echo "    still waiting for shared hostfile marker... (${_waited}s)"
    done
    _HOSTFILE="${_SHARED_HOSTFILE}"
    _IP_COL=1
    echo ">>> $(TZ=Asia/Shanghai date '+%H:%M:%S') Using shared hostfile: ${_HOSTFILE}"
fi

echo ">>> $(TZ=Asia/Shanghai date '+%H:%M:%S') Waiting for $_HOSTFILE to contain ${EXPECTED_NNODES} entries..."
_waited=0
while :; do
    _cur=$(awk 'END{print NR}' "$_HOSTFILE")
    if [[ "${_cur}" -ge "${EXPECTED_NNODES}" ]]; then
        echo ">>> $(TZ=Asia/Shanghai date '+%H:%M:%S') hostfile ready: ${_cur}/${EXPECTED_NNODES} lines."
        break
    fi
    if [[ ${_waited} -ge ${HOSTFILE_WAIT_SEC} ]]; then
        echo "ERROR: hostfile only has ${_cur}/${EXPECTED_NNODES} entries after ${HOSTFILE_WAIT_SEC}s. Aborting." >&2
        exit 1
    fi
    echo "    hostfile has ${_cur}/${EXPECTED_NNODES} lines, sleeping 5s... (${_waited}s waited)"
    sleep 5
    _waited=$((_waited + 5))
done

export MASTER_ADDR=$(awk -v c="$_IP_COL" 'NR==1{print $c; exit}' "$_HOSTFILE")
export NNODES="${EXPECTED_NNODES}"
export NODE_RANK=$(awk -v ip="${MY_NODE_IP:-}" -v c="$_IP_COL" -v n="$EXPECTED_NNODES" \
    'NR<=n && $c==ip{print NR-1; exit}' "$_HOSTFILE")

if [[ -z "${NODE_RANK}" ]]; then
    echo "ERROR: Cannot resolve NODE_RANK. MY_NODE_IP=${MY_NODE_IP:-<unset>} not in first ${EXPECTED_NNODES} lines of $_HOSTFILE." >&2
    awk -v c="$_IP_COL" -v n="$EXPECTED_NNODES" \
        'NR<=n{printf "  line %d: ip=%s\n", NR, $c}' "$_HOSTFILE" >&2
    exit 1
fi

: "${MASTER_PORT:=29500}"
TOTAL_GPU=$(( NNODES * NPROC_PER_NODE ))
DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-/tmp/fastwam_deepspeed_${RUN_ID}.hostfile}"
mapfile -t _DEEPSPEED_HOSTS < <(awk -v c="${_IP_COL}" -v n="${EXPECTED_NNODES}" '
    NR <= n {
        if ($c == "" || seen[$c]++) {
            exit 1
        }
        print $c
    }
    END {
        if (NR < n) {
            exit 1
        }
    }
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
if [[ ! -r "${DEEPSPEED_HOSTFILE}" ]]; then
    echo "ERROR: DeepSpeed hostfile is not readable: ${DEEPSPEED_HOSTFILE}" >&2
    exit 1
fi

# Extract gradient_accumulation_steps from Hydra overrides (if present), so
# effective global batch and the automatically scaled LR stay consistent.
MNODE_GRAD_ACCUM=1
for _arg in "$@"; do
    if [[ "${_arg}" == gradient_accumulation_steps=* ]]; then
        MNODE_GRAD_ACCUM="${_arg#gradient_accumulation_steps=}"
        break
    fi
done

GLOBAL_BATCH=$(( TOTAL_GPU * MNODE_PER_RANK_BATCH * MNODE_GRAD_ACCUM ))

: "${MNODE_LR_REF:=1e-4}"
: "${MNODE_LR_REF_BATCH:=1024}"
LR=$(awk -v b="${MNODE_LR_REF}" -v gb="${GLOBAL_BATCH}" -v ref="${MNODE_LR_REF_BATCH}" \
    'BEGIN{printf "%.4e", b * sqrt(gb / ref)}')

# ---------------------------------------------------------------------------
# NCCL / KCCL (跨节点通信加速)
# ---------------------------------------------------------------------------
_CUDA13_LIB="/usr/local/cuda-13.2/targets/x86_64-linux/lib"
: "${KCCL_LIB:=/opt/kccl/ubuntu/cuda13/libkccl.so.2.1-2.29.7.ubuntu-cuda13}"
: "${USE_KCCL:=0}"

# KCCL 仅提供 cuda13 版本。若 PyTorch 的 CUDA runtime 主版本不是 13,
# LD_PRELOAD cuda13 libs 会导致 "named symbol not found",必须跳过。
_TORCH_CUDA_MAJOR=$("${CONDA_ENV}/bin/python" -c "import torch,sys; sys.stdout.write((torch.version.cuda or '0').split('.')[0])" 2>/dev/null || echo "?")

if [[ "${USE_KCCL:-auto}" == "0" ]]; then
    echo ">>> KCCL disabled by USE_KCCL=0. Using native NCCL."
elif [[ "${_TORCH_CUDA_MAJOR}" != "13" ]]; then
    echo ">>> KCCL skipped: PyTorch CUDA=${_TORCH_CUDA_MAJOR} incompatible with KCCL (cuda13 only)."
    echo "    Falling back to native NCCL (still IB-RDMA, just less optimized)."
elif [[ -f "${KCCL_LIB}" ]] && [[ -d "${_CUDA13_LIB}" ]]; then
    export LD_PRELOAD="${KCCL_LIB}:${_CUDA13_LIB}/libcublasLt.so.13:${_CUDA13_LIB}/libcublas.so.13${LD_PRELOAD:+:${LD_PRELOAD}}"
    export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"   # KCCL 要求
    echo ">>> KCCL loaded: ${KCCL_LIB}"
else
    echo ">>> WARNING: KCCL not found (or cuda13 libs missing), falling back to native NCCL." >&2
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# A800 平台会在容器环境里注入 NCCL_IB_DISABLE=1, 导致 :-0 的 default 不生效, 通信全走
# socket over eth0 (TCP), 跨机吞吐骤降到 IB 的 1/5。这里给一个开关: FORCE_IB=1 时强制清掉
# NCCL_IB_DISABLE 让 IB/RoCE 正常工作。默认打开。
: "${FORCE_IB:=1}"
if [[ "${FORCE_IB}" == "1" ]]; then
    export NCCL_IB_DISABLE=0
    echo ">>> FORCE_IB=1: overriding NCCL_IB_DISABLE -> 0 (enable IB/RoCE for cross-node allreduce)"
fi
# A800 集群 mlx5 设备在部分节点报 RoCE、部分报 IB, AWS OFI NCCL plugin (ib_plugin.c)
# 无视 NCCL_IB_DISABLE 直接建 IB verbs 连接, 会出现:
#   NCCL WARN NET/IB : Remote IB device is incompatible with the local [0]mlx5_0:1/RoCE.
# 解决: 禁用 OFI 插件(回退到原生 NCCL socket/IB net), 并把 HCA 收敛到单一设备。
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"

# 自动探测最佳 IB HCA: 扫描所有 IB 设备, 选 active_mtu 最大的且网口 Up 的设备。
# 这样在不同批次机器上都能自动选到 400Gbps 高速网卡 (如 mlx5_bond_10),
# 而不是硬编码 mlx5_0 (在很多机器上不存在, 导致 NCCL 回退到慢速管理网)。
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
    _BEST_HCA=""
    _BEST_MTU=0
    if command -v ibdev2netdev &>/dev/null && command -v ibv_devinfo &>/dev/null; then
    while read -r _line; do
        # ibdev2netdev output: "mlx5_bond_0 port 1 ==> bond0 (Up)"
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
        echo ">>> Auto-detected NCCL_IB_HCA=${_BEST_HCA} (active_mtu=${_BEST_MTU}, highest among Up ports)"
    else
        echo ">>> WARNING: Could not auto-detect best IB HCA; leaving NCCL_IB_HCA unset (NCCL will auto-select)."
    fi
    unset _BEST_HCA _BEST_MTU _ibdev _mtu _line
else
    export NCCL_IB_HCA
    echo ">>> NCCL_IB_HCA=${NCCL_IB_HCA} (user-specified)"
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
    NCCL_SOCKET_IFNAME="$(ip route get "${_ROUTE_PEER}" 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "dev") {print $(i + 1); exit}}')"
    if [[ -z "${NCCL_SOCKET_IFNAME}" ]]; then
        echo "ERROR: Cannot resolve the NIC route to ${_ROUTE_PEER}; set NCCL_SOCKET_IFNAME explicitly." >&2
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

# ---------------------------------------------------------------------------
# 清理 MASTER_PORT 残留 (rank 0 才做)
# ---------------------------------------------------------------------------
if [[ "${NODE_RANK}" -eq 0 ]]; then
    echo ">>> $(TZ=Asia/Shanghai date '+%H:%M:%S') Cleaning stale processes on port ${MASTER_PORT}..."
    pkill -f "accelerate.*${MASTER_PORT}" 2>/dev/null || true
    pkill -f "torchrun.*${MASTER_PORT}"   2>/dev/null || true
    sleep 2
    _wait=0
    while ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT}\\b"; do
        [[ ${_wait} -ge 120 ]] && break
        echo "    port ${MASTER_PORT} still in use, waiting... (${_wait}s)"
        sleep 5
        _wait=$((_wait + 5))
    done
fi

# ---------------------------------------------------------------------------
# 数据/权重路径校验
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Auto-resume: 如果 output_dir 下有已有 checkpoint,自动传 resume=<latest state>
# ---------------------------------------------------------------------------
# 每台机器写各自的 log 文件,避免多节点并发 append 同一个 Ceph 文件行交错。
# rank 0 写 train.log(主日志,含 loss/step),其他节点写 train.node<rank>.log。
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

# ---------------------------------------------------------------------------
# 组装 hydra overrides
# ---------------------------------------------------------------------------
HYDRA_OVERRIDES=(
    "task=${TASK_NAME}"
    "batch_size=${MNODE_PER_RANK_BATCH}"
    "num_epochs=${MNODE_EPOCHS}"
    "learning_rate=${LR}"
    "save_every=${MNODE_SAVE_EVERY}"
    "eval_every=${MNODE_EVAL_EVERY}"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    "$@"
)

# ---------------------------------------------------------------------------
# 打印摘要
# ---------------------------------------------------------------------------
echo "========================================================================================================================"
echo "FastWAM RoboTwin MULTI-NODE training"
echo "  root              : ${FASTWAM_ROOT}"
echo "  hostfile          : ${_HOSTFILE}"
echo "  MY_NODE_IP        : ${MY_NODE_IP:-<unset>}"
echo "  MASTER_ADDR       : ${MASTER_ADDR}"
echo "  MASTER_PORT       : ${MASTER_PORT}"
echo "  NNODES            : ${NNODES}"
echo "  NODE_RANK         : ${NODE_RANK}"
echo "  NPROC_PER_NODE    : ${NPROC_PER_NODE}"
echo "  TOTAL_GPU         : ${TOTAL_GPU}"
echo "  RUN_ID            : ${RUN_ID}"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "  LOG_FILE          : ${LOG_FILE}"
echo "  epochs            : ${MNODE_EPOCHS}"
echo "  per-rank batch    : ${MNODE_PER_RANK_BATCH}"
echo "  grad_accum        : ${MNODE_GRAD_ACCUM}"
echo "  GLOBAL_BATCH      : ${GLOBAL_BATCH}"
echo "  learning_rate     : ${LR}"
echo "  deepspeed hostfile: ${DEEPSPEED_HOSTFILE}"
if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "  resume            : ${RESUME_ARGS[0]#resume=} (auto-detected)"
elif (( USER_RESUME_OVERRIDE == 1 )); then
    echo "  resume            : ${USER_RESUME_VALUE} (user-specified)"
else
    echo "  resume            : <none> (fresh training)"
fi
echo "========================================================================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] All preflight checks passed; not launching."
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
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
    "wandb.name=${TASK_NAME}" \
    "${HYDRA_OVERRIDES[@]}" 2>&1 | tee -a "${LOG_FILE}"
