# AIStudio Multi-Node hiWAM

This folder is isolated from the single-node training scripts. Use it for
AIStudio job submission and multi-node diagnostics only.

## 20-step 2-node H20 training test

```bash
AISTUDIO_COMMAND_MODE=train \
AISTUDIO_CACHE_KIND=shm \
WORKER_NUM=1 \
GPUS_PER_NODE=8 \
MAX_STEPS=20 \
PER_GPU_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=1 \
/team/xinda.qi/envs/fastwam/bin/python scripts/aistudio_multinode/submit_aistudio_hiwam.py
```

Expected log markers:

```text
[aistudio_local_ckpt] PREWARM_BEGIN
[prewarm_resolve] dit: ... files=3
[prewarm_resolve] vae: ... files=1
[prewarm] total_mib=...
[aistudio_local_ckpt] PREWARM_DONE
[aistudio_local_ckpt] TRAIN_LAUNCH_BEGIN
[launch] nproc_per_node=8 total_processes=16 num_machines=2
Loading Wan2.2-TI2V-5B components...
epoch=0 step=10/20
epoch=0 step=20/20
[aistudio_local_ckpt] TRAIN_LAUNCH_DONE return_code=0
```

If the log stops before `TRAIN_LAUNCH_BEGIN`, the job only completed staging or
prewarm and did not run training.
