#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/team/xinda.qi/envs/fastwam/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN (${PYTHON_BIN}) is not executable." >&2
  exit 1
fi

FASTWAM_ENV="${FASTWAM_ENV:-$(dirname "$(dirname "${PYTHON_BIN}")")}"
SOURCE_CKPT_DIR="${FASTWAM_SOURCE_CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"
LOCAL_CKPT_DIR="${FASTWAM_LOCAL_CHECKPOINT_DIR:-/tmp/hiwam_checkpoints}"

echo "[aistudio_local_ckpt] host=$(hostname)"
echo "[aistudio_local_ckpt] source_ckpt=${SOURCE_CKPT_DIR}"
echo "[aistudio_local_ckpt] local_ckpt=${LOCAL_CKPT_DIR}"
df -h /tmp "${SOURCE_CKPT_DIR}" 2>/dev/null || true

bash "${SCRIPT_DIR}/../stage_checkpoints_local.sh" "${SOURCE_CKPT_DIR}" "${LOCAL_CKPT_DIR}"

export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_CKPT_DIR}"
export FASTWAM_OUTPUT_ROOT="${FASTWAM_OUTPUT_ROOT:-runs/aistudio_multinode}"
echo "[aistudio_local_ckpt] DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "[aistudio_local_ckpt] FASTWAM_OUTPUT_ROOT=${FASTWAM_OUTPUT_ROOT}"
echo "[aistudio_local_ckpt] launching train_zero1_aistudio.sh"

bash "${SCRIPT_DIR}/train_zero1_aistudio.sh" "$@"
