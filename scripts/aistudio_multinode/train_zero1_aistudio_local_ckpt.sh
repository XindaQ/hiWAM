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
export FASTWAM_ENV
export PATH="${FASTWAM_ENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export FASTWAM_STAGE_DEREFERENCE_SYMLINKS="${FASTWAM_STAGE_DEREFERENCE_SYMLINKS:-0}"
export FASTWAM_MATERIALIZE_CHECKPOINTS="${FASTWAM_MATERIALIZE_CHECKPOINTS:-1}"

echo "[aistudio_local_ckpt] host=$(hostname)"
echo "[aistudio_local_ckpt] source_ckpt=${SOURCE_CKPT_DIR}"
echo "[aistudio_local_ckpt] local_ckpt=${LOCAL_CKPT_DIR}"
echo "[aistudio_local_ckpt] python=${PYTHON_BIN}"
echo "[aistudio_local_ckpt] dereference_symlinks=${FASTWAM_STAGE_DEREFERENCE_SYMLINKS}"
echo "[aistudio_local_ckpt] materialize_checkpoints=${FASTWAM_MATERIALIZE_CHECKPOINTS}"
echo "[aistudio_local_ckpt] hydra_overrides=${*:2}"
mkdir -p "$(dirname "${LOCAL_CKPT_DIR}")"
df -h / /tmp /dev/shm "$(dirname "${LOCAL_CKPT_DIR}")" "${SOURCE_CKPT_DIR}" 2>/dev/null || true

bash "${SCRIPT_DIR}/../stage_checkpoints_local.sh" "${SOURCE_CKPT_DIR}" "${LOCAL_CKPT_DIR}"

export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_CKPT_DIR}"
if [[ "${FASTWAM_MATERIALIZE_CHECKPOINTS}" != "0" ]]; then
  echo "[aistudio_local_ckpt] MATERIALIZE_BEGIN"
  bash "${SCRIPT_DIR}/materialize_checkpoints.sh" "${DIFFSYNTH_MODEL_BASE_PATH}" "${@:2}"
  echo "[aistudio_local_ckpt] MATERIALIZE_DONE"
else
  echo "[aistudio_local_ckpt] checkpoint materialize disabled"
fi

if [[ "${FASTWAM_PREWARM_CHECKPOINTS:-1}" != "0" ]]; then
  echo "[aistudio_local_ckpt] PREWARM_BEGIN"
  bash "${SCRIPT_DIR}/prewarm_checkpoints.sh" "${DIFFSYNTH_MODEL_BASE_PATH}" "${@:2}"
  echo "[aistudio_local_ckpt] PREWARM_DONE"
else
  echo "[aistudio_local_ckpt] checkpoint prewarm disabled"
fi

export FASTWAM_OUTPUT_ROOT="${FASTWAM_OUTPUT_ROOT:-runs/aistudio_multinode}"
echo "[aistudio_local_ckpt] DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "[aistudio_local_ckpt] FASTWAM_OUTPUT_ROOT=${FASTWAM_OUTPUT_ROOT}"
echo "[aistudio_local_ckpt] TRAIN_LAUNCH_BEGIN"

set +e
bash "${SCRIPT_DIR}/train_zero1_aistudio.sh" "$@"
train_rc=$?
set -e
echo "[aistudio_local_ckpt] TRAIN_LAUNCH_DONE return_code=${train_rc}"
exit "${train_rc}"
