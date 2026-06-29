#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/team/xinda.qi/envs/fastwam/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN (${PYTHON_BIN}) is not executable." >&2
  exit 1
fi

FASTWAM_ENV="${FASTWAM_ENV:-$(dirname "$(dirname "${PYTHON_BIN}")")}"
TARGET_ROOT="${FASTWAM_FFMPEG_ROOT:-${FASTWAM_ENV}/ffmpeg7}"
TARGET_LIB="${TARGET_ROOT}/lib"
SOURCE="${1:-}"

scan_dirs=()
if [[ -n "${SOURCE}" ]]; then
  if [[ ! -d "${SOURCE}" ]]; then
    echo "Error: source directory does not exist: ${SOURCE}" >&2
    exit 1
  fi
  scan_dirs+=("${SOURCE}")
else
  scan_dirs+=(
    "${FASTWAM_ENV}/lib"
    /usr/lib64
    /usr/local/lib64
    /usr/lib
    /usr/local/lib
    /opt/conda/lib
  )
fi

find_first() {
  local pattern="$1"
  local dir
  for dir in "${scan_dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    find "${dir}" -maxdepth 1 -name "${pattern}" -type f -print -quit
  done
}

declare -A libs=(
  [libavutil.so.59]="libavutil.so.59*"
  [libavcodec.so.61]="libavcodec.so.61*"
  [libavformat.so.61]="libavformat.so.61*"
  [libswscale.so.8]="libswscale.so.8*"
  [libswresample.so.5]="libswresample.so.5*"
)

TMP_LIB="${TARGET_LIB}.tmp.$(hostname).$$"
rm -rf "${TMP_LIB}"
mkdir -p "${TMP_LIB}"

echo "[stage_ffmpeg] target=${TARGET_LIB}"
echo "[stage_ffmpeg] scan=${scan_dirs[*]}"

missing=0
for soname in "${!libs[@]}"; do
  src="$(find_first "${libs[${soname}]}")"
  if [[ -z "${src}" ]]; then
    echo "[stage_ffmpeg] missing ${soname} (${libs[${soname}]})" >&2
    missing=1
    continue
  fi
  base="$(basename "${src}")"
  cp -a "${src}" "${TMP_LIB}/${base}"
  ln -sf "${base}" "${TMP_LIB}/${soname}"
  echo "[stage_ffmpeg] ${soname} <- ${src}"
done

if (( missing != 0 )); then
  rm -rf "${TMP_LIB}"
  cat >&2 <<'EOF'
Error: FFmpeg 7 shared libraries were not found.

Install or expose FFmpeg 7 libs in the current container, then rerun:
  bash scripts/stage_ffmpeg_libs_nas.sh

Or pass a directory that already contains libavutil.so.59, libavcodec.so.61,
libavformat.so.61, libswscale.so.8, and libswresample.so.5:
  bash scripts/stage_ffmpeg_libs_nas.sh /path/to/ffmpeg7/lib
EOF
  exit 1
fi

mkdir -p "${TARGET_ROOT}"
rm -rf "${TARGET_LIB}.old"
if [[ -d "${TARGET_LIB}" ]]; then
  mv "${TARGET_LIB}" "${TARGET_LIB}.old"
fi
mv "${TMP_LIB}" "${TARGET_LIB}"

echo "[stage_ffmpeg] validating torchcodec"
LD_LIBRARY_PATH="${TARGET_LIB}:${FASTWAM_ENV}/lib:${LD_LIBRARY_PATH:-}" "${PYTHON_BIN}" - <<'PY'
from torchcodec.decoders import VideoDecoder
print("torchcodec ok")
PY

echo "[stage_ffmpeg] done"
echo "[stage_ffmpeg] FASTWAM_FFMPEG_LIB_DIR=${TARGET_LIB}"
