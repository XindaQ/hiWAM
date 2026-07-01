#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: bash scripts/stage_checkpoints_local.sh <source_dir> <target_dir>}"
TARGET_DIR="${2:?Usage: bash scripts/stage_checkpoints_local.sh <source_dir> <target_dir>}"
READY_MARKER="${TARGET_DIR}/.fastwam_stage_complete"
DEREFERENCE_SYMLINKS="${FASTWAM_STAGE_DEREFERENCE_SYMLINKS:-0}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Error: source checkpoint directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -f "${READY_MARKER}" ]]; then
  marker_mode="$(sed -n 's/^dereference_symlinks=//p' "${READY_MARKER}" 2>/dev/null | tail -n 1)"
  marker_mode="${marker_mode:-0}"
  if [[ "${marker_mode}" == "${DEREFERENCE_SYMLINKS}" ]]; then
    echo "[stage] using existing local checkpoints: ${TARGET_DIR}"
    echo "[stage] dereference_symlinks=${DEREFERENCE_SYMLINKS}"
    exit 0
  fi
  echo "Error: target is marked complete with a different staging mode: ${TARGET_DIR}" >&2
  echo "Expected dereference_symlinks=${DEREFERENCE_SYMLINKS}; found dereference_symlinks=${marker_mode}" >&2
  echo "Marker contents:" >&2
  cat "${READY_MARKER}" >&2 || true
  echo "Use a new empty target directory, or inspect/remove this directory yourself." >&2
  exit 1
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
echo "[stage] dereference_symlinks=${DEREFERENCE_SYMLINKS}"
source_symlink_count="$(find "${SOURCE_DIR}" -type l | wc -l | tr -d ' ')"
echo "[stage] source_symlink_count=${source_symlink_count}"

SECONDS=0
if [[ "${DEREFERENCE_SYMLINKS}" == "1" ]]; then
  tar -h -C "${SOURCE_DIR}" -cf - . | tar -C "${TMP_DIR}" -xf -
else
  tar -C "${SOURCE_DIR}" -cf - . | tar -C "${TMP_DIR}" -xf -
fi
copy_seconds="${SECONDS}"
target_symlink_count="$(find "${TMP_DIR}" -type l | wc -l | tr -d ' ')"
echo "[stage] target_symlink_count=${target_symlink_count}"

printf 'dereference_symlinks=%s\n' "${DEREFERENCE_SYMLINKS}" > "${TMP_DIR}/.fastwam_stage_complete"
rmdir "${TARGET_DIR}" 2>/dev/null || true
mv "${TMP_DIR}" "${TARGET_DIR}"

echo "[stage] done seconds=${copy_seconds}"
du -sh "${TARGET_DIR}" || true
