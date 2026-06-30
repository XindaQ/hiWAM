#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT="${1:?Usage: bash scripts/aistudio_multinode/prewarm_checkpoints.sh <checkpoint_root>}"
PREWARM_FILES="${FASTWAM_PREWARM_FILES:-Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors,Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors,Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
CHUNK_MB="${FASTWAM_PREWARM_CHUNK_MB:-64}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[prewarm] host=$(hostname)"
echo "[prewarm] root=${CHECKPOINT_ROOT}"
echo "[prewarm] files=${PREWARM_FILES}"
echo "[prewarm] chunk_mb=${CHUNK_MB}"
echo "[prewarm] python=${PYTHON_BIN}"
df -h "${CHECKPOINT_ROOT}" / /tmp /dev/shm 2>/dev/null || true

"${PYTHON_BIN}" - "${CHECKPOINT_ROOT}" "${PREWARM_FILES}" "${CHUNK_MB}" <<'PY'
from pathlib import Path
import socket
import sys
import time

root = Path(sys.argv[1])
relative_files = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
chunk_mb = int(sys.argv[3])
chunk_size = chunk_mb * 1024 * 1024

if not root.is_dir():
    print(f"[prewarm] skip: checkpoint root does not exist: {root}", flush=True)
    raise SystemExit(2)

def is_readable_candidate(path: Path) -> bool:
    return path.exists() or path.is_symlink()


files = []
missing = []
for relative_file in relative_files:
    path = root / relative_file
    if is_readable_candidate(path):
        files.append(path)
        print(f"[prewarm] file_check path={path} exists=1", flush=True)
    else:
        missing.append(path)
        print(f"[prewarm] file_check path={path} exists=0", flush=True)

if missing:
    print("[prewarm] missing required checkpoint files", flush=True)
    preview = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.exists() or path.is_symlink()
    )[:50]
    for path in missing:
        print(f"[prewarm] missing_file={path}", flush=True)
    for item in preview:
        print(f"[prewarm] available_file_preview={item}", flush=True)
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
