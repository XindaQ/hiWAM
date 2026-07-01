#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${1:?Usage: bash scripts/aistudio_multinode/prewarm_checkpoints.sh <checkpoint_root>}"
shift
CHUNK_MB="${FASTWAM_PREWARM_CHUNK_MB:-64}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HYDRA_OVERRIDES=("$@")

echo "[prewarm] host=$(hostname)"
echo "[prewarm] root=${CHECKPOINT_ROOT}"
echo "[prewarm] hydra_overrides=${HYDRA_OVERRIDES[*]:-<none>}"
echo "[prewarm] chunk_mb=${CHUNK_MB}"
echo "[prewarm] python=${PYTHON_BIN}"
df -h "${CHECKPOINT_ROOT}" / /tmp /dev/shm 2>/dev/null || true

RESOLVED_FILE="$(mktemp /tmp/hiwam_prewarm_files.XXXXXX)"
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
echo "[prewarm] RESOLVE_DONE resolved_file_count=${resolved_count}"
while IFS= read -r resolved_path; do
    [[ -n "${resolved_path}" ]] && echo "[prewarm] resolved_path=${resolved_path}"
done < "${RESOLVED_FILE}"

"${PYTHON_BIN}" - "${RESOLVED_FILE}" "${CHUNK_MB}" <<'PY'
from pathlib import Path
import socket
import sys
import time

file_list_path = Path(sys.argv[1])
chunk_mb = int(sys.argv[2])
chunk_size = chunk_mb * 1024 * 1024

files = []
missing = []
for line in file_list_path.read_text().splitlines():
    if not line.strip():
        continue
    path = Path(line.strip())
    if path.exists() or path.is_symlink():
        files.append(path)
        print(f"[prewarm] file_check path={path} exists=1", flush=True)
    else:
        missing.append(path)
        print(f"[prewarm] file_check path={path} exists=0", flush=True)

if missing:
    print("[prewarm] missing required checkpoint files", flush=True)
    for path in missing:
        print(f"[prewarm] missing_file={path}", flush=True)
    raise SystemExit(2)

if not files:
    print("[prewarm] no checkpoint files resolved", flush=True)
    raise SystemExit(2)

total_bytes = 0
total_seconds = 0.0
checksum = 0
print(f"[prewarm] python_host={socket.gethostname()} files={len(files)}", flush=True)

for path in files:
    if path.is_symlink():
        try:
            target = path.readlink()
        except OSError as exc:
            target = f"<readlink_failed:{exc}>"
        print(f"[prewarm] symlink={path} target={target}", flush=True)
    size_bytes = path.stat().st_size
    size_mib = size_bytes // (1024 * 1024)
    print(f"[prewarm] file={path} size_mib={size_mib}", flush=True)
    start = time.perf_counter()
    with path.open("rb", buffering=0) as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            checksum ^= data[0]
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        elapsed = 0.001
    mib_per_s = size_mib / elapsed
    print(
        f"[prewarm] done file={path} seconds={elapsed:.3f} "
        f"approx_mib_per_s={mib_per_s:.2f}",
        flush=True,
    )
    total_bytes += size_bytes
    total_seconds += elapsed

total_mib = total_bytes // (1024 * 1024)
if total_seconds <= 0:
    total_seconds = 0.001
print(
    f"[prewarm] total_mib={total_mib} total_seconds={total_seconds:.3f} "
    f"approx_mib_per_s={total_mib / total_seconds:.2f} checksum={checksum}",
    flush=True,
)
PY
