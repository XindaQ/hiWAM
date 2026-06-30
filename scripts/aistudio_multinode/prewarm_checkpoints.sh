#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${1:?Usage: bash scripts/aistudio_multinode/prewarm_checkpoints.sh <checkpoint_root>}"
PATTERN="${FASTWAM_PREWARM_PATTERN:-Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors}"
CHUNK_MB="${FASTWAM_PREWARM_CHUNK_MB:-64}"

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "[prewarm] skip: checkpoint root does not exist: ${CHECKPOINT_ROOT}"
  exit 0
fi

echo "[prewarm] host=$(hostname)"
echo "[prewarm] root=${CHECKPOINT_ROOT}"
echo "[prewarm] pattern=${PATTERN}"
echo "[prewarm] chunk_mb=${CHUNK_MB}"
df -h "${CHECKPOINT_ROOT}" / /tmp /dev/shm 2>/dev/null || true

mapfile -d '' files < <(find "${CHECKPOINT_ROOT}" -path "${CHECKPOINT_ROOT}/${PATTERN}" -type f -print0 | sort -z)
if (( ${#files[@]} == 0 )); then
  echo "[prewarm] no files matched"
  exit 0
fi

total_bytes=0
total_seconds=0
for file in "${files[@]}"; do
  size_bytes="$(stat -c%s "${file}")"
  size_mib=$((size_bytes / 1024 / 1024))
  echo "[prewarm] file=${file} size_mib=${size_mib}"
  SECONDS=0
  dd if="${file}" of=/dev/null bs="${CHUNK_MB}M" iflag=fullblock status=none
  seconds="${SECONDS}"
  if (( seconds <= 0 )); then
    seconds=1
  fi
  mib_per_s=$((size_mib / seconds))
  echo "[prewarm] done file=${file} seconds=${SECONDS} approx_mib_per_s=${mib_per_s}"
  total_bytes=$((total_bytes + size_bytes))
  total_seconds=$((total_seconds + SECONDS))
done

total_mib=$((total_bytes / 1024 / 1024))
if (( total_seconds <= 0 )); then
  total_seconds=1
fi
echo "[prewarm] total_mib=${total_mib} total_seconds=${total_seconds} approx_mib_per_s=$((total_mib / total_seconds))"
