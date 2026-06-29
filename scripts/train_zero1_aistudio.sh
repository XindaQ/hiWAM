#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1_aistudio.sh <nproc_per_node> [hydra_overrides...]}"
shift

PYTHON_BIN="${PYTHON_BIN:-/team/xinda.qi/envs/fastwam/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN (${PYTHON_BIN}) is not executable. Set PYTHON_BIN to the intended environment python." >&2
  exit 1
fi
FASTWAM_ENV="${FASTWAM_ENV:-$(dirname "$(dirname "${PYTHON_BIN}")")}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-${FASTWAM_ENV}/bin/deepspeed}"
export FASTWAM_ENV DEEPSPEED_BIN
export PATH="${FASTWAM_ENV}/bin:${PATH}"
FASTWAM_FFMPEG_LIB_DIR="${FASTWAM_FFMPEG_LIB_DIR:-${FASTWAM_ENV}/ffmpeg7/lib}"
if [[ -d "${FASTWAM_FFMPEG_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FASTWAM_FFMPEG_LIB_DIR}:${FASTWAM_ENV}/lib:${LD_LIBRARY_PATH:-}"
  echo "[launch] ffmpeg_lib=${FASTWAM_FFMPEG_LIB_DIR}"
else
  export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib:${LD_LIBRARY_PATH:-}"
fi
FFMPEG_BIN="$("${PYTHON_BIN}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
if [[ -n "${FFMPEG_BIN}" && -x "${FFMPEG_BIN}" ]]; then
  mkdir -p /tmp/hiwam_bin
  ln -sf "${FFMPEG_BIN}" /tmp/hiwam_bin/ffmpeg
  export PATH="/tmp/hiwam_bin:${PATH}"
  export IMAGEIO_FFMPEG_EXE="${FFMPEG_BIN}"
  echo "[launch] ffmpeg=$(command -v ffmpeg)"
fi
if [[ "${FASTWAM_TORCHCODEC_CHECK:-warn}" != "off" ]]; then
  CHECK_ARGS=()
  if [[ "${FASTWAM_TORCHCODEC_CHECK:-warn}" == "strict" ]]; then
    CHECK_ARGS+=(--strict)
  fi
  "${PYTHON_BIN}" scripts/check_torchcodec_ffmpeg.py "${CHECK_ARGS[@]}"
fi
if [[ ! -x "${DEEPSPEED_BIN}" ]]; then
  echo "Error: DEEPSPEED_BIN (${DEEPSPEED_BIN}) is not executable." >&2
  exit 1
fi
export WANDB_DIR="${WANDB_DIR:-/team/xinda.qi/project-zhou/wandb}"
mkdir -p "${WANDB_DIR}"
EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}), NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

TOTAL_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))
DEEPSPEED_MULTINODE_LAUNCHER="${DEEPSPEED_MULTINODE_LAUNCHER:-standard}"

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      "${PYTHON_BIN}" - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} total_processes=${TOTAL_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} main_process=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} run_id=${RUN_ID} deepspeed_multinode_launcher=${DEEPSPEED_MULTINODE_LAUNCHER}"
echo "[launch] python_bin=${PYTHON_BIN} fastwam_env=${FASTWAM_ENV} deepspeed_bin=${DEEPSPEED_BIN} path_deepspeed=$(command -v deepspeed || true)"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --deepspeed_multinode_launcher "${DEEPSPEED_MULTINODE_LAUNCHER}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
