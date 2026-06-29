#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: bash scripts/stage_checkpoints_local.sh <source_dir> <target_dir>}"
TARGET_DIR="${2:?Usage: bash scripts/stage_checkpoints_local.sh <source_dir> <target_dir>}"
READY_MARKER="${TARGET_DIR}/.fastwam_stage_complete"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Error: source checkpoint directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -f "${READY_MARKER}" ]]; then
  echo "[stage] using existing local checkpoints: ${TARGET_DIR}"
  exit 0
fi

if [[ -d "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Error: target exists but is not marked complete: ${TARGET_DIR}" >&2
  echo "Use a new empty target directory, or inspect/remove this partial directory yourself." >&2
  exit 1
fi

TMP_DIR="${TARGET_DIR}.tmp.$(hostname).$$"
if [[ -e "${TMP_DIR}" ]]; then
  echo "Error: temporary staging path already exists: ${TMP_DIR}" >&2
  exit 1
fi

mkdir -p "${TMP_DIR}"
echo "[stage] copying checkpoints safely"
echo "[stage] source=${SOURCE_DIR}"
echo "[stage] target=${TARGET_DIR}"
echo "[stage] tmp=${TMP_DIR}"

SECONDS=0
tar -C "${SOURCE_DIR}" -cf - . | tar -C "${TMP_DIR}" -xf -
copy_seconds="${SECONDS}"

touch "${TMP_DIR}/.fastwam_stage_complete"
rmdir "${TARGET_DIR}" 2>/dev/null || true
mv "${TMP_DIR}" "${TARGET_DIR}"

echo "[stage] done seconds=${copy_seconds}"
du -sh "${TARGET_DIR}" || true
