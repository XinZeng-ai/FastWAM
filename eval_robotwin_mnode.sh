#!/usr/bin/env bash
set -euo pipefail

# Run the identical command on every node. Evaluation workers are independent;
# nodes coordinate only through the shared FastWAM filesystem.

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${FASTWAM_ROOT}"

: "${EXPECTED_NNODES:=2}"
: "${GPUS_PER_NODE:=8}"
: "${WORKERS_PER_GPU:=2}"
: "${HOSTFILE_WAIT_SEC:=600}"
: "${MNODE_COORD_TIMEOUT_SEC:=3600}"
: "${HOSTFILE_START_INDEX:=0}"
: "${EVAL_RUN_ID:?EVAL_RUN_ID is required for multi-node evaluation}"
: "${MNODE_EVAL_SESSION_ID:=${EVAL_RUN_ID}}"

if ! [[ "${EXPECTED_NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXPECTED_NNODES must be a positive integer, got: ${EXPECTED_NNODES}" >&2
  exit 2
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GPUS_PER_NODE must be a positive integer, got: ${GPUS_PER_NODE}" >&2
  exit 2
fi
if ! [[ "${HOSTFILE_START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: HOSTFILE_START_INDEX must be a non-negative integer, got: ${HOSTFILE_START_INDEX}" >&2
  exit 2
fi

# NODE_RANK may be supplied explicitly. Otherwise use the same platform
# hostfile discovery convention as the multi-node training launchers.
if [[ -z "${NODE_RANK:-}" ]]; then
  _COORD_DIR="${FASTWAM_ROOT}/evaluate_results/robotwin/.mnode_coord/${EVAL_RUN_ID}"
  mkdir -p "${_COORD_DIR}"
  _SHARED_HOSTFILE="${_COORD_DIR}/hostfile"
  _SHARED_READY="${_SHARED_HOSTFILE}.ready"

  if [[ -f /etc/kml/ssh_configmap/pod_list ]]; then
    _HOSTFILE=/etc/kml/ssh_configmap/pod_list
    _IP_COL=3
  elif [[ -f /etc/mpi/hostfile ]]; then
    _HOSTFILE=/etc/mpi/hostfile
    _IP_COL=1
  else
    echo ">>> Waiting for a shared hostfile: ${_SHARED_HOSTFILE}"
    _waited=0
    while [[ ! -f "${_SHARED_READY}" || ! -f "${_SHARED_HOSTFILE}" ]]; do
      if (( _waited >= HOSTFILE_WAIT_SEC )); then
        echo "ERROR: timeout waiting for shared hostfile (${HOSTFILE_WAIT_SEC}s)." >&2
        exit 1
      fi
      sleep 5
      _waited=$((_waited + 5))
    done
    _HOSTFILE="${_SHARED_HOSTFILE}"
    _IP_COL=1
  fi

  _required_lines=$((HOSTFILE_START_INDEX + EXPECTED_NNODES))
  echo ">>> Waiting for ${_HOSTFILE} to contain ${_required_lines} entries..."
  _waited=0
  while :; do
    _cur=$(awk 'END{print NR}' "${_HOSTFILE}")
    if (( _cur >= _required_lines )); then
      break
    fi
    if (( _waited >= HOSTFILE_WAIT_SEC )); then
      echo "ERROR: hostfile has ${_cur}/${_required_lines} entries after ${HOSTFILE_WAIT_SEC}s." >&2
      exit 1
    fi
    sleep 5
    _waited=$((_waited + 5))
  done

  # Publish a normalized one-column hostfile for nodes whose platform mount
  # does not expose the original hostfile.
  _tmp_hostfile="${_SHARED_HOSTFILE}.tmp.$$"
  awk -v c="${_IP_COL}" -v start="${HOSTFILE_START_INDEX}" -v n="${EXPECTED_NNODES}" \
    'NR > start && NR <= start + n {print $c}' "${_HOSTFILE}" > "${_tmp_hostfile}"
  mv -f "${_tmp_hostfile}" "${_SHARED_HOSTFILE}"
  : > "${_SHARED_READY}"

  NODE_RANK=$(awk -v ip="${MY_NODE_IP:-}" \
    '$1 == ip {print NR-1; exit}' "${_SHARED_HOSTFILE}")
  if [[ -z "${NODE_RANK}" ]]; then
    echo "ERROR: cannot resolve NODE_RANK; MY_NODE_IP=${MY_NODE_IP:-<unset>} is not in ${_SHARED_HOSTFILE}." >&2
    exit 1
  fi
fi

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= EXPECTED_NNODES )); then
  echo "ERROR: NODE_RANK must be in [0, ${EXPECTED_NNODES}), got: ${NODE_RANK}" >&2
  exit 2
fi

export NNODES="${EXPECTED_NNODES}"
export NODE_RANK
export NUM_GPUS="${GPUS_PER_NODE}"
export WORKERS_PER_GPU
export MNODE_EVAL_SESSION_ID
export MNODE_COORD_TIMEOUT_SEC

echo "===================================================================================================="
echo "FastWAM RoboTwin MULTI-NODE evaluation"
echo "  NNODES                 : ${NNODES}"
echo "  NODE_RANK              : ${NODE_RANK}"
echo "  GPUS_PER_NODE          : ${GPUS_PER_NODE}"
echo "  WORKERS_PER_GPU        : ${WORKERS_PER_GPU}"
echo "  TOTAL_GPUS             : $((NNODES * GPUS_PER_NODE))"
echo "  TOTAL_WORKERS          : $((NNODES * GPUS_PER_NODE * WORKERS_PER_GPU))"
echo "  TASK_CONFIG            : ${TASK_CONFIG:-<default>}"
echo "  CKPT_PATH              : ${CKPT_PATH:-<default>}"
echo "  EVAL_RUN_ID            : ${EVAL_RUN_ID}"
echo "  MNODE_EVAL_SESSION_ID  : ${MNODE_EVAL_SESSION_ID}"
echo "===================================================================================================="

exec bash "${FASTWAM_ROOT}/eval_robotwin.sh"
