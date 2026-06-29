#!/usr/bin/env python3
"""Minimal multi-node NCCL communication smoke test."""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", default="1,16,64,256,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def main() -> None:
    args = parse_args()
    local_rank = env_int("LOCAL_RANK")
    rank = env_int("RANK")
    world_size = env_int("WORLD_SIZE", 1)
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(
            "COMM_SMOKE_START "
            f"host={socket.gethostname()} world_size={world_size} "
            f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"nccl={torch.cuda.nccl.version() if torch.cuda.is_available() else None}",
            flush=True,
        )
    print(
        "COMM_SMOKE_RANK "
        f"rank={rank} local_rank={local_rank} host={socket.gethostname()} "
        f"cuda_device={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )
    dist.barrier()

    for size_mb in [int(item) for item in args.sizes_mb.split(",") if item.strip()]:
        numel = size_mb * 1024 * 1024 // 4
        tensor = torch.zeros(numel, device=device, dtype=torch.float32)

        for _ in range(args.warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed * 1000.0 / args.iters
        # Approximate bus bandwidth for all-reduce: 2 * (world_size - 1) / world_size payload.
        bus_gbps = (size_mb / 1024.0) * (2.0 * (world_size - 1) / world_size) / (avg_ms / 1000.0)
        if rank == 0:
            print(
                "COMM_SMOKE_RESULT "
                f"size_mb={size_mb} avg_ms={avg_ms:.3f} approx_bus_GBps={bus_gbps:.3f} "
                f"iters={args.iters} world_size={world_size}",
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        print("COMM_SMOKE_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
