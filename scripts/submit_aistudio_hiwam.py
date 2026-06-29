#!/usr/bin/env python3
"""Submit a hiWAM/FastWAM multi-node GPU training job to AIStudio.

This submitter intentionally keeps the project training configs unchanged.
Training-specific knobs are passed as Hydra overrides in `TRAIN_OVERRIDES`.
"""

from pypai.conf import ExecConf, GpuType, KMConf
from pypai.job import PythonJobBuilder


# AIStudio environment.
IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:13880163-20250915220702"
K8S_APP_NAME = "agenth20"
CLUSTER = "auto"
KM_POOL = "kubemaker"
NAS_MOUNT_POINT = "/team"
NAS_EXPORT = "26d2d249ad1-jnj31.cn-heyuan-alipay.nas.aliyuncs.com:/"
PROJECT_DIR = "/team/xinda.qi/project-zhou/code/hiWAM"
PYTHON_BIN = "/team/xinda.qi/envs/fastwam/bin/python"
WANDB_DIR = "/team/xinda.qi/project-zhou/wandb"
LOG_DIR = "/team/xinda.qi/project-zhou/aistudio_job_logs"

# Use "smoke" first to verify AIStudio can start the 2-node/16-GPU job.
# Switch to "train" after the simple command succeeds.
COMMAND_MODE = "smoke"

# Resource shape. For 2 nodes x 8 GPUs, keep WORKER_NUM = 1.
# For 8 nodes x 8 GPUs, set WORKER_NUM = 7.
GPUS_PER_NODE = 8
WORKER_NUM = 1
GPU_TYPE = GpuType.H20
CPU_PER_NODE = 128
MEMORY_MB_PER_NODE = 1572864
DISK_MB_PER_NODE = 1638400

# Keep the original config files untouched; override runtime knobs here.
TRAIN_SCRIPT = "scripts/train_zero1.sh"
TRAIN_OVERRIDES = [
    "task=libero_uncond_2cam224_1e-4",
    "batch_size=8",
    "max_steps=20",
    "num_workers=4",
    "wandb.mode=offline",
    "output_dir=/team/xinda.qi/project-zhou/runs/aistudio_hiwam_smoke_2n8g_b8_s20",
]


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_logged_command(mode: str, commands: list[str]) -> str:
    log_file = (
        f"{shell_quote(LOG_DIR)}/hiwam_{mode}_rank_${{RANK:-unknown}}_"
        "$(hostname)_$(date +%Y%m%d_%H%M%S).log"
    )
    prelude = [
        "set -eo pipefail",
        f"mkdir -p {shell_quote(NAS_MOUNT_POINT)}",
        f"(mountpoint -q {shell_quote(NAS_MOUNT_POINT)} || mount -t nfs -o vers=3,nolock,proto=tcp {shell_quote(NAS_EXPORT)} {shell_quote(NAS_MOUNT_POINT)})",
        f"mkdir -p {shell_quote(LOG_DIR)}",
        f"LOG_FILE={log_file}",
        'exec > >(tee -a "$LOG_FILE") 2>&1',
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
    ])


def build_train_command() -> str:
    overrides = " ".join(shell_quote(item) for item in TRAIN_OVERRIDES)
    return build_logged_command("train", [
        "export SSL_NO_VERIFY=1",
        "printenv",
        f"ls -ld {shell_quote(PROJECT_DIR)}",
        f"ls -l {shell_quote(PYTHON_BIN)}",
        f"cd {shell_quote(PROJECT_DIR)}",
        f"export PYTHON_BIN={shell_quote(PYTHON_BIN)}",
        f"export WANDB_DIR={shell_quote(WANDB_DIR)}",
        "export NCCL_DEBUG=INFO",
        "export NNODES=${WORLD_SIZE}",
        "export NODE_RANK=${RANK}",
        "unset RANK WORLD_SIZE LOCAL_RANK",
        f"bash {shell_quote(TRAIN_SCRIPT)} {GPUS_PER_NODE} {overrides}",
    ])


def build_command() -> str:
    if COMMAND_MODE == "smoke":
        return build_smoke_command()
    if COMMAND_MODE == "train":
        return build_train_command()
    raise ValueError(f"Unsupported COMMAND_MODE: {COMMAND_MODE}")


def main():
    print("[submit] image:", IMAGE)
    print("[submit] app:", K8S_APP_NAME)
    print("[submit] cluster:", CLUSTER)
    print("[submit] gpu_type:", GPU_TYPE)
    print("[submit] command_mode:", COMMAND_MODE)
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
        host_network=True,
        k8s_app_name=K8S_APP_NAME,
        k8s_priority="low",
        tag="type=SFT,basemodel=Wan2.2-TI2V-5B",
        platform="kubemaker",
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    main()
