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
NAS_MOUNT_POINT = "/team"
NAS_EXPORT = "26d2d249ad1-jnj31.cn-heyuan-alipay.nas.aliyuncs.com:/"
PROJECT_DIR = "/team/xinda.qi/project-zhou/hiWAM"
PYTHON_BIN = "/team/xinda.qi/envs/fastwam/bin/python"
WANDB_DIR = "/team/xinda.qi/project-zhou/wandb"

# Resource shape. For 2 nodes x 8 GPUs, keep WORKER_NUM = 1.
# For 8 nodes x 8 GPUs, set WORKER_NUM = 7.
GPUS_PER_NODE = 8
WORKER_NUM = 1
GPU_TYPE = GpuType.A100
CPU_PER_NODE = 64
MEMORY_MB_PER_NODE = 500000
DISK_MB_PER_NODE = 102400

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


def build_command() -> str:
    overrides = " \\\n  ".join(shell_quote(item) for item in TRAIN_OVERRIDES)
    return f"""set -euo pipefail

mkdir -p {shell_quote(NAS_MOUNT_POINT)}
if ! grep -qs " {NAS_MOUNT_POINT} " /proc/mounts; then
  mount -t nfs -o vers=3,nolock,proto=tcp {shell_quote(NAS_EXPORT)} {shell_quote(NAS_MOUNT_POINT)}
fi
export SSL_NO_VERIFY=1

cd {shell_quote(PROJECT_DIR)}

export PYTHON_BIN={shell_quote(PYTHON_BIN)}
export WANDB_DIR={shell_quote(WANDB_DIR)}
export NCCL_DEBUG="${{NCCL_DEBUG:-INFO}}"

# AIStudio exposes node-level WORLD_SIZE/RANK. hiWAM scripts expect NNODES/NODE_RANK.
export NNODES="${{WORLD_SIZE}}"
export NODE_RANK="${{RANK}}"

# Avoid leaking AIStudio's node-level rank names into libraries that expect process ranks.
unset RANK WORLD_SIZE LOCAL_RANK

bash {shell_quote(TRAIN_SCRIPT)} {GPUS_PER_NODE} \\
  {overrides}
"""


def main():
    km_conf = KMConf(image=IMAGE)
    master = ExecConf(
        num=1,
        cpu=CPU_PER_NODE,
        memory=MEMORY_MB_PER_NODE,
        gpu_num=GPUS_PER_NODE,
        gpu_type=GPU_TYPE,
        disk_m=DISK_MB_PER_NODE,
    )
    worker = ExecConf(
        num=WORKER_NUM,
        cpu=CPU_PER_NODE,
        memory=MEMORY_MB_PER_NODE,
        gpu_num=GPUS_PER_NODE,
        gpu_type=GPU_TYPE,
        disk_m=DISK_MB_PER_NODE,
    )

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
        k8s_priority="high",
        tag="type=SFT,basemodel=Wan2.2-TI2V-5B",
        platform="kubemaker",
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    main()
