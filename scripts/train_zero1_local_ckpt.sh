#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export FASTWAM_STAGE_DEREFERENCE_SYMLINKS="${FASTWAM_STAGE_DEREFERENCE_SYMLINKS:-0}"
export FASTWAM_MATERIALIZE_PREWARM_FILES="${FASTWAM_MATERIALIZE_PREWARM_FILES:-1}"

if [[ -n "${FASTWAM_LOCAL_CHECKPOINT_DIR:-}" ]]; then
  LOCAL_CKPT_DIR="${FASTWAM_LOCAL_CHECKPOINT_DIR}"
elif [[ "${FASTWAM_MATERIALIZE_PREWARM_FILES}" == "1" ]]; then
  LOCAL_CKPT_DIR="/tmp/hiwam_checkpoints_materialized"
else
  LOCAL_CKPT_DIR="/tmp/hiwam_checkpoints"
fi
SOURCE_CKPT_DIR="${FASTWAM_SOURCE_CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"
DEBUG_ROOT="${FASTWAM_LOAD_DEBUG_DIR:-${PROJECT_DIR}/runs/load_debug}"
DEBUG_LOG="${DEBUG_ROOT}/$(date +%Y%m%d_%H%M%S)_local_ckpt.log"

mkdir -p "${DEBUG_ROOT}"
exec > >(tee -a "${DEBUG_LOG}") 2>&1

echo "[debug] log=${DEBUG_LOG}"
echo "[debug] host=$(hostname)"
echo "[debug] date=$(date)"
echo "[debug] source_ckpt=${SOURCE_CKPT_DIR}"
echo "[debug] local_ckpt=${LOCAL_CKPT_DIR}"
echo "[debug] dereference_symlinks=${FASTWAM_STAGE_DEREFERENCE_SYMLINKS}"
echo "[debug] materialize_prewarm_files=${FASTWAM_MATERIALIZE_PREWARM_FILES}"
echo "[debug] disk"
df -h /tmp /workspace /ossfs /team 2>/dev/null || true
echo "[debug] source_size"
du -sh "${SOURCE_CKPT_DIR}" || true

bash "${SCRIPT_DIR}/stage_checkpoints_local.sh" "${SOURCE_CKPT_DIR}" "${LOCAL_CKPT_DIR}"

export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_CKPT_DIR}"
if [[ "${FASTWAM_MATERIALIZE_PREWARM_FILES}" == "1" ]]; then
  echo "[debug] PREWARM_MATERIALIZE_BEGIN"
  bash "${SCRIPT_DIR}/aistudio_multinode/prewarm_checkpoints.sh" "${DIFFSYNTH_MODEL_BASE_PATH}" "${@:2}"
  echo "[debug] PREWARM_MATERIALIZE_DONE"
fi
echo "[debug] DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "[debug] launching original train_zero1.sh"

bash "${SCRIPT_DIR}/train_zero1.sh" "$@"
