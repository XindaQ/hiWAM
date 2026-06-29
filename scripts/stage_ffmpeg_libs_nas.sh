#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/team/xinda.qi/envs/fastwam/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN (${PYTHON_BIN}) is not executable." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
  AV_LIBS="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import pathlib

try:
    import av
except Exception:
    raise SystemExit(0)

libdir = pathlib.Path(av.__file__).resolve().parent.parent / "av.libs"
if libdir.is_dir():
    print(libdir)
PY
)"
  if [[ -n "${AV_LIBS}" && -d "${AV_LIBS}" ]]; then
    scan_dirs+=("${AV_LIBS}")
  fi
fi

find_first() {
  local soname="$1"
  local base="${soname%%.so.*}"
  local suffix="${soname#${base}.so.}"
  local dir
  for dir in "${scan_dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    find "${dir}" -maxdepth 1 \( -name "${soname}*" -o -name "${base}"'-*.so.'"${suffix}"'*' \) -type f -print -quit
  done
}

version_sonames() {
  case "$1" in
    7)
      printf '%s\n' libavutil.so.59 libavcodec.so.61 libavformat.so.61 libavdevice.so.61 libavfilter.so.10 libswscale.so.8 libswresample.so.5
      ;;
    6)
      printf '%s\n' libavutil.so.58 libavcodec.so.60 libavformat.so.60 libavdevice.so.60 libavfilter.so.9 libswscale.so.7 libswresample.so.4
      ;;
    5)
      printf '%s\n' libavutil.so.57 libavcodec.so.59 libavformat.so.59 libavdevice.so.59 libavfilter.so.8 libswscale.so.6 libswresample.so.4
      ;;
    4)
      printf '%s\n' libavutil.so.56 libavcodec.so.58 libavformat.so.58 libavdevice.so.58 libavfilter.so.7 libswscale.so.5 libswresample.so.3
      ;;
    *)
      return 1
      ;;
  esac
}

TMP_LIB="${TARGET_LIB}.tmp.$(hostname).$$"
rm -rf "${TMP_LIB}"
mkdir -p "${TMP_LIB}"

echo "[stage_ffmpeg] target=${TARGET_LIB}"
echo "[stage_ffmpeg] scan=${scan_dirs[*]}"

selected_version=""
selected_sonames=()
selected_sources=()

for version in 7 6 5 4; do
  current_sonames=()
  current_sources=()
  found_all=1
  while IFS= read -r soname; do
    src="$(find_first "${soname}")"
    if [[ -z "${src}" ]]; then
      found_all=0
      break
    fi
    current_sonames+=("${soname}")
    current_sources+=("${src}")
  done < <(version_sonames "${version}")

  if (( found_all == 1 )); then
    selected_version="${version}"
    selected_sonames=("${current_sonames[@]}")
    selected_sources=("${current_sources[@]}")
    break
  fi
done

if [[ -z "${selected_version}" ]]; then
  rm -rf "${TMP_LIB}"
  cat >&2 <<'EOF'
Error: supported FFmpeg shared libraries were not found.

TorchCodec supports FFmpeg 4, 5, 6, and 7. Install or expose one complete
set of FFmpeg shared libraries in the current container, then rerun:
  bash scripts/stage_ffmpeg_libs_nas.sh

Or pass a directory that already contains one complete supported set:
  bash scripts/stage_ffmpeg_libs_nas.sh /path/to/ffmpeg/lib
EOF
  exit 1
fi

echo "[stage_ffmpeg] selected_ffmpeg_major=${selected_version}"

copied_bundle_dirs=()
for idx in "${!selected_sonames[@]}"; do
  soname="${selected_sonames[${idx}]}"
  src="${selected_sources[${idx}]}"
  src_dir="$(dirname "${src}")"
  base="$(basename "${src}")"
  cp -a "${src}" "${TMP_LIB}/${base}"
  ln -sf "${base}" "${TMP_LIB}/${soname}"
  echo "[stage_ffmpeg] ${soname} <- ${src}"

  if [[ "$(basename "${src_dir}")" == "av.libs" ]]; then
    already_copied=0
    for copied_dir in "${copied_bundle_dirs[@]}"; do
      if [[ "${copied_dir}" == "${src_dir}" ]]; then
        already_copied=1
        break
      fi
    done
    if (( already_copied == 0 )); then
      echo "[stage_ffmpeg] bundled_deps <- ${src_dir}/lib*.so*"
      find "${src_dir}" -maxdepth 1 -name 'lib*.so*' -type f -exec cp -a {} "${TMP_LIB}/" \;
      copied_bundle_dirs+=("${src_dir}")
    fi
  fi
done

mkdir -p "${TARGET_ROOT}"
rm -rf "${TARGET_LIB}.old"
if [[ -d "${TARGET_LIB}" ]]; then
  mv "${TARGET_LIB}" "${TARGET_LIB}.old"
fi
mv "${TMP_LIB}" "${TARGET_LIB}"

echo "[stage_ffmpeg] validating torchcodec"
LD_LIBRARY_PATH="${TARGET_LIB}:${FASTWAM_ENV}/lib:${LD_LIBRARY_PATH:-}" "${PYTHON_BIN}" "${SCRIPT_DIR}/check_torchcodec_ffmpeg.py" --strict --prefix "[stage_ffmpeg][torchcodec]"

echo "[stage_ffmpeg] done"
echo "[stage_ffmpeg] FASTWAM_FFMPEG_LIB_DIR=${TARGET_LIB}"
