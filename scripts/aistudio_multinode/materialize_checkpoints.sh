#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${1:?Usage: bash scripts/aistudio_multinode/materialize_checkpoints.sh <checkpoint_root>}"
shift
CHUNK_MB="${FASTWAM_MATERIALIZE_CHUNK_MB:-${FASTWAM_PREWARM_CHUNK_MB:-64}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HYDRA_OVERRIDES=("$@")

echo "[materialize_stage] host=$(hostname)"
echo "[materialize_stage] root=${CHECKPOINT_ROOT}"
echo "[materialize_stage] hydra_overrides=${HYDRA_OVERRIDES[*]:-<none>}"
echo "[materialize_stage] chunk_mb=${CHUNK_MB}"
echo "[materialize_stage] python=${PYTHON_BIN}"
df -h "${CHECKPOINT_ROOT}" / /tmp /dev/shm 2>/dev/null || true

RESOLVED_FILE="$(mktemp /tmp/hiwam_materialize_files.XXXXXX)"
cleanup() {
    rm -f "${RESOLVED_FILE}"
}
trap cleanup EXIT

if [[ -n "${FASTWAM_PREWARM_FILES:-}" ]]; then
    "${PYTHON_BIN}" - "${CHECKPOINT_ROOT}" "${FASTWAM_PREWARM_FILES}" > "${RESOLVED_FILE}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
for item in sys.argv[2].split(","):
    item = item.strip()
    if not item:
        continue
    path = Path(item)
    if not path.is_absolute():
        path = root / path
    print(path)
PY
else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/resolve_prewarm_files.py" \
        --checkpoint-root "${CHECKPOINT_ROOT}" \
        --project-root "${PROJECT_DIR}" \
        "${HYDRA_OVERRIDES[@]}" > "${RESOLVED_FILE}"
fi

resolved_count="$(wc -l < "${RESOLVED_FILE}" | tr -d ' ')"
echo "[materialize_stage] RESOLVE_DONE resolved_file_count=${resolved_count}"
while IFS= read -r resolved_path; do
    [[ -n "${resolved_path}" ]] && echo "[materialize_stage] resolved_path=${resolved_path}"
done < "${RESOLVED_FILE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_checkpoint_files.py" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --file-list "${RESOLVED_FILE}" \
    --chunk-mb "${CHUNK_MB}"

echo "[materialize_stage] DONE"
