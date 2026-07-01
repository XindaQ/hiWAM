#!/usr/bin/env python3
"""Submit a hiWAM/FastWAM multi-node GPU training job to AIStudio.

This submitter intentionally keeps the project training configs unchanged.
Training-specific knobs are passed as Hydra overrides in `TRAIN_OVERRIDES`.
"""

from datetime import datetime
import os
from uuid import uuid4

from pypai.conf import ExecConf, GpuType, KMConf
from pypai.job import PythonJobBuilder


# AIStudio environment.
IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:13880163-20250915220702"
K8S_APP_NAME = "agenth20"
CLUSTER = os.environ.get("AISTUDIO_CLUSTER", "auto")
KM_POOL = "kubemaker"
NAS_MOUNT_POINT = "/team"
NAS_EXPORT = "26d2d249ad1-jnj31.cn-heyuan-alipay.nas.aliyuncs.com:/"
PROJECT_DIR = "/team/xinda.qi/project-zhou/code/hiWAM"
PYTHON_BIN = "/team/xinda.qi/envs/fastwam/bin/python"
WANDB_DIR = "/team/xinda.qi/project-zhou/wandb"
LOG_ROOT = "/team/xinda.qi/project-zhou/aistudio_job_logs"
CACHE_KIND = os.environ.get("AISTUDIO_CACHE_KIND", "tmp").strip().lower()
if CACHE_KIND not in {"tmp", "shm"}:
    raise ValueError("AISTUDIO_CACHE_KIND must be either 'tmp' or 'shm'.")
LOCAL_CHECKPOINT_DIR = os.environ.get(
    "AISTUDIO_LOCAL_CHECKPOINT_DIR",
    "/dev/shm/hiwam_checkpoints" if CACHE_KIND == "shm" else "/tmp/hiwam_checkpoints",
)

# Use "smoke" to verify AIStudio can start all nodes; "rdma" for lightweight RDMA
# diagnostics; "comm" for NCCL all-reduce diagnostics; "loadbench" for
# safetensors checkpoint-load profiling; use "train" for the 20-step training
# test. macOS may set COMMAND_MODE=unix2003,
# so prefer the AIStudio-specific name and only accept known legacy values.
_LEGACY_COMMAND_MODE = os.environ.get("COMMAND_MODE")
COMMAND_MODE = os.environ.get("AISTUDIO_COMMAND_MODE")
if COMMAND_MODE is None:
    COMMAND_MODE = _LEGACY_COMMAND_MODE if _LEGACY_COMMAND_MODE in {"smoke", "rdma", "comm", "loadbench", "train"} else "train"

# Resource shape. For 2 nodes x 8 GPUs, keep WORKER_NUM = 1.
# For 8 nodes x 8 GPUs, set WORKER_NUM = 7.
# RDMA diagnostics can override these to a tiny shape, e.g. GPUS_PER_NODE=1.
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "8"))
WORKER_NUM = int(os.environ.get("WORKER_NUM", "1"))
NODE_COUNT = WORKER_NUM + 1
TOTAL_GPUS = NODE_COUNT * GPUS_PER_NODE
GPU_TYPE = GpuType.H20
CPU_PER_NODE = int(os.environ.get("CPU_PER_NODE", "128"))
MEMORY_MB_PER_NODE = int(os.environ.get("MEMORY_MB_PER_NODE", "1572864"))
DISK_MB_PER_NODE = int(os.environ.get("DISK_MB_PER_NODE", "1638400"))

# Keep the original config files untouched; override runtime knobs here.
TRAIN_SCRIPT = os.environ.get(
    "TRAIN_SCRIPT",
    "scripts/aistudio_multinode/train_zero1_aistudio_local_ckpt.sh",
)
COMM_SCRIPT = "scripts/aistudio_multinode/nccl_comm_smoke.py"
TRAIN_TASK = os.environ.get("TRAIN_TASK", "libero_idm_2cam224_1e-4")
PER_GPU_BATCH_SIZE = int(os.environ.get("PER_GPU_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1"))
MAX_STEPS = os.environ.get("MAX_STEPS", "20").strip()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
ZERO_STAGE = 1
COMM_SIZES_MB = os.environ.get("COMM_SIZES_MB", "1,16,64,256,512")
COMM_WARMUP = int(os.environ.get("COMM_WARMUP", "5"))
COMM_ITERS = int(os.environ.get("COMM_ITERS", "20"))
LOAD_BENCH_METHODS = os.environ.get(
    "LOAD_BENCH_METHODS",
    "raw_read,safe_to_dtype_hold,safe_to_dtype_hold,safe_get_hold,safe_to_dtype_discard,load_file_mmap_to_dtype,load_file_pread_to_dtype",
)
LOAD_BENCH_MAX_FILES = os.environ.get("LOAD_BENCH_MAX_FILES", "0")
LOAD_BENCH_TORCH_THREADS = os.environ.get("LOAD_BENCH_TORCH_THREADS", "").strip()
FASTWAM_PREWARM_CHECKPOINTS = os.environ.get("FASTWAM_PREWARM_CHECKPOINTS", "1")
FASTWAM_PREWARM_CHUNK_MB = os.environ.get("FASTWAM_PREWARM_CHUNK_MB", "64")
FASTWAM_SERIALIZE_WAN_LOAD = os.environ.get("FASTWAM_SERIALIZE_WAN_LOAD", "0")
if "uncond" in TRAIN_TASK and os.environ.get("ALLOW_UNCOND", "0") != "1":
    raise ValueError(
        f"Refusing to submit uncond task by default: {TRAIN_TASK}. "
        "Set ALLOW_UNCOND=1 only if this is intentional."
    )
RUN_TAG = (
    f"{datetime.now():%Y%m%d_%H%M%S}_"
    f"{COMMAND_MODE}_"
    f"{NODE_COUNT}n{TOTAL_GPUS}g_"
    f"bs{PER_GPU_BATCH_SIZE}_acc{GRADIENT_ACCUMULATION_STEPS}_"
    f"cache{CACHE_KIND}_"
    f"s{MAX_STEPS if MAX_STEPS else 'full'}_z{ZERO_STAGE}_"
    f"{uuid4().hex[:6]}"
)
LOG_DIR = f"{LOG_ROOT}/{RUN_TAG}"
TRAIN_OVERRIDES = [
    f"task={TRAIN_TASK}",
    f"batch_size={PER_GPU_BATCH_SIZE}",
    f"gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS}",
    f"num_workers={NUM_WORKERS}",
    "wandb.mode=offline",
    f"wandb.name={RUN_TAG}",
]
if MAX_STEPS:
    TRAIN_OVERRIDES.append(f"max_steps={MAX_STEPS}")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_logged_command(mode: str, commands: list[str]) -> str:
    log_file = f"{shell_quote(LOG_DIR)}/rank_${{RANK:-unknown}}.log"
    prelude = [
        "set -eo pipefail",
        f"mkdir -p {shell_quote(NAS_MOUNT_POINT)}",
        f"(mountpoint -q {shell_quote(NAS_MOUNT_POINT)} || mount -t nfs -o vers=3,nolock,proto=tcp {shell_quote(NAS_EXPORT)} {shell_quote(NAS_MOUNT_POINT)})",
        f"mkdir -p {shell_quote(LOG_DIR)}",
        f"LOG_FILE={log_file}",
        'exec > >(tee -a "$LOG_FILE") 2>&1',
        f"echo RUN_TAG={shell_quote(RUN_TAG)}",
        f"echo LOG_DIR={shell_quote(LOG_DIR)}",
        'echo LOG_FILE="$LOG_FILE"',
        "set -x",
    ]
    return "bash -lc " + shell_quote("; ".join(prelude + commands))


def build_smoke_command() -> str:
    pycheck = (
        "import sys, torch, accelerate, deepspeed; "
        "print('PYTHON', sys.executable); "
        "print('TORCH', torch.__version__); "
        "print('CUDA_AVAILABLE', torch.cuda.is_available()); "
        "print('CUDA_DEVICE_COUNT', torch.cuda.device_count()); "
        "print('ACCELERATE', accelerate.__version__); "
        "print('DEEPSPEED', deepspeed.__version__)"
    )
    return build_logged_command("smoke", [
        "echo AI_ENV MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} WORLD_SIZE=${WORLD_SIZE:-} RANK=${RANK:-}",
        "hostname",
        "date",
        "nvidia-smi",
        "mount | grep ' /team ' || true",
        f"ls -ld {shell_quote(PROJECT_DIR)}",
        f"ls -l {shell_quote(PYTHON_BIN)}",
        f"{shell_quote(PYTHON_BIN)} -c {shell_quote(pycheck)}",
        "echo SMOKE_DONE_SLEEPING_TO_KEEP_ALL_NODES_ALIVE",
        "sleep 60",
    ])


def build_rdma_command() -> str:
    pycheck = (
        "import torch; "
        "print('torch', torch.__version__); "
        "print('cuda_available', torch.cuda.is_available()); "
        "print('cuda_device_count', torch.cuda.device_count()); "
        "print('nccl_version', torch.cuda.nccl.version() if torch.cuda.is_available() else None)"
    )
    return build_logged_command("rdma", [
        "export SSL_NO_VERIFY=1",
        "echo AI_ENV MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} WORLD_SIZE=${WORLD_SIZE:-} RANK=${RANK:-}",
        "hostname",
        "date",
        "nvidia-smi",
        "nvidia-smi topo -m || true",
        "echo '===== rpm rdma packages ====='",
        "echo '$ rpm -qa | grep -i rdma'",
        "(rpm -qa | grep -i rdma) || echo 'NO_OUTPUT: rpm -qa | grep -i rdma'",
        "echo '===== libmlx5 / ibverbs libraries ====='",
        "echo '$ ldconfig -p | grep -i libmlx5'",
        "(ldconfig -p 2>/dev/null | grep -i libmlx5) || echo 'NO_OUTPUT: ldconfig -p | grep -i libmlx5'",
        "echo '$ ldconfig -p | grep -Ei libibverbs/libnccl/libmlx'",
        "(ldconfig -p 2>/dev/null | grep -Ei 'libibverbs|libnccl|libmlx') || echo 'NO_OUTPUT: ldconfig -p | grep -Ei libibverbs/libnccl/libmlx'",
        "echo '===== infiniband devices ====='",
        "ls -l /dev/infiniband || true",
        "echo '$ ibv_devinfo'",
        "(command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo) || echo 'COMMAND_NOT_FOUND_OR_FAILED: ibv_devinfo'",
        "echo '===== nccl env ====='",
        "echo '$ env | grep -i nccl'",
        "(env | grep -i nccl) || echo 'NO_OUTPUT: env | grep -i nccl'",
        "echo '===== network interfaces ====='",
        "ip -br addr || true",
        "echo '===== python/torch/nccl versions ====='",
        f"cd {shell_quote(PROJECT_DIR)}",
        f"export PYTHON_BIN={shell_quote(PYTHON_BIN)}",
        f"{shell_quote(PYTHON_BIN)} -c {shell_quote(pycheck)}",
        "echo RDMA_DIAG_DONE_SLEEPING_TO_KEEP_ALL_NODES_ALIVE",
        "sleep 60",
    ])


def build_comm_command() -> str:
    return build_logged_command("comm", [
        "export SSL_NO_VERIFY=1",
        "echo AI_ENV MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} WORLD_SIZE=${WORLD_SIZE:-} RANK=${RANK:-}",
        "hostname",
        "date",
        "env | sort | grep -E '^(NCCL|UCX|IB|CUDA|MASTER|WORLD_SIZE|RANK|HOST_PORTS|NODE_|POD_|ALIPAY_APP_IDC|AISTUDIO_PROJECT_NAME)=' || true",
        "nvidia-smi",
        "nvidia-smi topo -m || true",
        "ls -l /dev/infiniband || true",
        "(command -v ibv_devinfo && ibv_devinfo) || true",
        "(command -v ibstat && ibstat) || true",
        "ldconfig -p 2>/dev/null | grep -E 'ibverbs|nccl|mlx' || true",
        "find /usr /opt /team/xinda.qi/envs/fastwam -name 'libnccl-net.so*' -o -name 'libibverbs.so*' 2>/dev/null | head -50 || true",
        f"ls -ld {shell_quote(PROJECT_DIR)}",
        f"ls -l {shell_quote(PYTHON_BIN)}",
        f"cd {shell_quote(PROJECT_DIR)}",
        f"export PYTHON_BIN={shell_quote(PYTHON_BIN)}",
        'export FASTWAM_ENV="$(dirname "$(dirname "$PYTHON_BIN")")"',
        'unset PYTHONPATH PythonPath',
        'export PYTHONPATH="$PWD/src"',
        'export PATH="$FASTWAM_ENV/bin:$PATH"',
        'export LD_LIBRARY_PATH="$FASTWAM_ENV/lib:${LD_LIBRARY_PATH:-}"',
        'export VIRTUAL_ENV="$FASTWAM_ENV"',
        'export CONDA_PREFIX="$FASTWAM_ENV"',
        'export CONDA_DEFAULT_ENV="$(basename "$FASTWAM_ENV")"',
        "export PYTHONNOUSERSITE=1",
        "hash -r",
        "export NCCL_DEBUG=INFO",
        "export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,COLL",
        "export NNODES=${WORLD_SIZE}",
        "export NODE_RANK=${RANK}",
        "unset RANK WORLD_SIZE LOCAL_RANK",
        (
            f"{shell_quote(PYTHON_BIN)} -m torch.distributed.run "
            f"--nnodes \"${{NNODES}}\" "
            f"--nproc_per_node {GPUS_PER_NODE} "
            f"--node_rank \"${{NODE_RANK}}\" "
            f"--master_addr \"${{MASTER_ADDR}}\" "
            f"--master_port \"${{MASTER_PORT}}\" "
            f"{shell_quote(COMM_SCRIPT)} "
            f"--sizes-mb {shell_quote(COMM_SIZES_MB)} "
            f"--warmup {COMM_WARMUP} "
            f"--iters {COMM_ITERS}"
        ),
    ])


def build_train_command() -> str:
    overrides = " ".join(shell_quote(item) for item in TRAIN_OVERRIDES)
    return build_logged_command("train", [
        "export SSL_NO_VERIFY=1",
        "echo AI_ENV MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} WORLD_SIZE=${WORLD_SIZE:-} RANK=${RANK:-}",
        "hostname",
        "date",
        "nvidia-smi",
        "mount | grep ' /team ' || true",
        f"ls -ld {shell_quote(PROJECT_DIR)}",
        f"ls -l {shell_quote(PYTHON_BIN)}",
        f"cd {shell_quote(PROJECT_DIR)}",
        f"export PYTHON_BIN={shell_quote(PYTHON_BIN)}",
        f"export WANDB_DIR={shell_quote(WANDB_DIR)}",
        f"export FASTWAM_SOURCE_CHECKPOINT_DIR={shell_quote(PROJECT_DIR + '/checkpoints')}",
        f"export FASTWAM_LOCAL_CHECKPOINT_DIR={shell_quote(LOCAL_CHECKPOINT_DIR)}",
        f"export AISTUDIO_CACHE_KIND={shell_quote(CACHE_KIND)}",
        f"export FASTWAM_PREWARM_CHECKPOINTS={shell_quote(FASTWAM_PREWARM_CHECKPOINTS)}",
        f"export FASTWAM_PREWARM_CHUNK_MB={shell_quote(FASTWAM_PREWARM_CHUNK_MB)}",
        "export FASTWAM_OUTPUT_ROOT=runs/aistudio_multinode",
        "export FASTWAM_DEBUG_WAN_LOAD=1",
        f"export FASTWAM_SERIALIZE_WAN_LOAD={shell_quote(FASTWAM_SERIALIZE_WAN_LOAD)}",
        "export DIFFSYNTH_SKIP_DOWNLOAD=true",
        "export NCCL_DEBUG=WARN",
        "export RUN_ID_SYNC_TIMEOUT=1800",
        f"export RUN_ID={shell_quote(RUN_TAG)}",
        "export NNODES=${WORLD_SIZE}",
        "export NODE_RANK=${RANK}",
        "unset RANK WORLD_SIZE LOCAL_RANK",
        f"bash {shell_quote(TRAIN_SCRIPT)} {GPUS_PER_NODE} {overrides}",
    ])


def build_loadbench_command() -> str:
    return build_logged_command("loadbench", [
        "export SSL_NO_VERIFY=1",
        "echo AI_ENV MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} WORLD_SIZE=${WORLD_SIZE:-} RANK=${RANK:-}",
        "hostname",
        "date",
        "nvidia-smi",
        "mount | grep ' /team ' || true",
        f"ls -ld {shell_quote(PROJECT_DIR)}",
        f"ls -l {shell_quote(PYTHON_BIN)}",
        f"cd {shell_quote(PROJECT_DIR)}",
        f"export PYTHON_BIN={shell_quote(PYTHON_BIN)}",
        f"export FASTWAM_SOURCE_CHECKPOINT_DIR={shell_quote(PROJECT_DIR + '/checkpoints')}",
        f"export FASTWAM_LOCAL_CHECKPOINT_DIR={shell_quote(LOCAL_CHECKPOINT_DIR)}",
        f"export AISTUDIO_CACHE_KIND={shell_quote(CACHE_KIND)}",
        "export DIFFSYNTH_SKIP_DOWNLOAD=true",
        f"export LOAD_BENCH_METHODS={shell_quote(LOAD_BENCH_METHODS)}",
        f"export LOAD_BENCH_MAX_FILES={shell_quote(LOAD_BENCH_MAX_FILES)}",
        f"export LOAD_BENCH_TORCH_THREADS={shell_quote(LOAD_BENCH_TORCH_THREADS)}",
        "bash scripts/stage_checkpoints_local.sh \"${FASTWAM_SOURCE_CHECKPOINT_DIR}\" \"${FASTWAM_LOCAL_CHECKPOINT_DIR}\"",
        "export DIFFSYNTH_MODEL_BASE_PATH=\"${FASTWAM_LOCAL_CHECKPOINT_DIR}\"",
        "export FASTWAM_ENV=\"$(dirname \"$(dirname \"$PYTHON_BIN\")\")\"",
        "unset PYTHONPATH PythonPath",
        "export PYTHONPATH=\"$PWD/src\"",
        "export PATH=\"$FASTWAM_ENV/bin:/tmp/hiwam_bin:$PATH\"",
        "export LD_LIBRARY_PATH=\"/team/xinda.qi/envs/fastwam/ffmpeg7/lib:$FASTWAM_ENV/lib:${LD_LIBRARY_PATH:-}\"",
        "export VIRTUAL_ENV=\"$FASTWAM_ENV\"",
        "export CONDA_PREFIX=\"$FASTWAM_ENV\"",
        "export PYTHONNOUSERSITE=1",
        "hash -r",
        f"{shell_quote(PYTHON_BIN)} scripts/aistudio_multinode/safetensors_load_bench.py "
        "--checkpoint-root \"${DIFFSYNTH_MODEL_BASE_PATH}\" "
        "--dtype bf16 "
        "--methods \"${LOAD_BENCH_METHODS}\" "
        "--max-files \"${LOAD_BENCH_MAX_FILES}\" "
        "--torch-threads \"${LOAD_BENCH_TORCH_THREADS:-0}\"",
        "echo LOADBENCH_DONE_SLEEPING_TO_KEEP_ALL_NODES_ALIVE",
        "sleep 60",
    ])


def build_command() -> str:
    if COMMAND_MODE == "smoke":
        return build_smoke_command()
    if COMMAND_MODE == "rdma":
        return build_rdma_command()
    if COMMAND_MODE == "comm":
        return build_comm_command()
    if COMMAND_MODE == "loadbench":
        return build_loadbench_command()
    if COMMAND_MODE == "train":
        return build_train_command()
    raise ValueError(f"Unsupported COMMAND_MODE: {COMMAND_MODE}")


def main():
    print("[submit] image:", IMAGE)
    print("[submit] app:", K8S_APP_NAME)
    print("[submit] cluster:", CLUSTER)
    print("[submit] gpu_type:", GPU_TYPE)
    print("[submit] command_mode:", COMMAND_MODE)
    print("[submit] run_tag:", RUN_TAG)
    print("[submit] log_dir:", LOG_DIR)
    print("[submit] train_task:", TRAIN_TASK)
    print("[submit] per_gpu_batch_size:", PER_GPU_BATCH_SIZE)
    print("[submit] gradient_accumulation_steps:", GRADIENT_ACCUMULATION_STEPS)
    print("[submit] max_steps:", MAX_STEPS)
    print("[submit] train_script:", TRAIN_SCRIPT)
    print("[submit] cache_kind:", CACHE_KIND)
    print("[submit] local_checkpoint_dir:", LOCAL_CHECKPOINT_DIR)
    print("[submit] prewarm_checkpoints:", FASTWAM_PREWARM_CHECKPOINTS)
    print("[submit] prewarm_chunk_mb:", FASTWAM_PREWARM_CHUNK_MB)
    print("[submit] serialize_wan_load:", FASTWAM_SERIALIZE_WAN_LOAD)
    if COMMAND_MODE == "comm":
        print("[submit] comm_sizes_mb:", COMM_SIZES_MB)
        print("[submit] comm_warmup:", COMM_WARMUP)
        print("[submit] comm_iters:", COMM_ITERS)
    if COMMAND_MODE == "loadbench":
        print("[submit] load_bench_methods:", LOAD_BENCH_METHODS)
        print("[submit] load_bench_max_files:", LOAD_BENCH_MAX_FILES)
        print("[submit] load_bench_torch_threads:", LOAD_BENCH_TORCH_THREADS or "default")
    print("[submit] master: num=1 gpu_num=%s cpu=%s memory=%s disk_m=%s" % (
        GPUS_PER_NODE,
        CPU_PER_NODE,
        MEMORY_MB_PER_NODE,
        DISK_MB_PER_NODE,
    ))
    print("[submit] worker: num=%s gpu_num=%s cpu=%s memory=%s disk_m=%s" % (
        WORKER_NUM,
        GPUS_PER_NODE,
        CPU_PER_NODE,
        MEMORY_MB_PER_NODE,
        DISK_MB_PER_NODE,
    ))
    print("[submit] command:\n" + build_command())

    km_conf = KMConf(image=IMAGE, pool=KM_POOL, cluster=CLUSTER)
    exec_kwargs = dict(
        cpu=CPU_PER_NODE,
        memory=MEMORY_MB_PER_NODE,
        gpu_num=GPUS_PER_NODE,
        disk_m=DISK_MB_PER_NODE,
    )
    exec_kwargs["gpu_type"] = GPU_TYPE

    master = ExecConf(num=1, **exec_kwargs)
    worker = ExecConf(num=WORKER_NUM, **exec_kwargs)

    job = PythonJobBuilder(
        source_root=None,
        main_file="",
        command=build_command(),
        km_conf=km_conf,
        master=master,
        worker=worker,
        runtime="pytorch",
        rdma=True,
        hostNetwork=True,
        k8s_app_name=K8S_APP_NAME,
        k8s_priority="high",
        tag="type=SFT,basemodel=Wan2.2-TI2V-5B",
        platform="kubemaker",
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    main()
