#!/usr/bin/env python3
"""Benchmark Wan2.2 safetensors loading without accelerate/deepspeed.

This script is intentionally standalone. It measures the pieces that can make
the first Wan2.2 checkpoint touch slow inside an AIStudio pod:

* raw sequential file reads
* safe_open/get_tensor without dtype conversion
* safe_open/get_tensor plus dtype conversion, matching the project loader
* safetensors.torch.load_file with mmap/pread backends when supported
"""

from __future__ import annotations

import argparse
import gc
import glob
import inspect
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "checkpoints"))
    parser.add_argument("--pattern", default="Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors")
    parser.add_argument("--dtype", choices=["none", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--methods", default=os.environ.get("LOAD_BENCH_METHODS", "safe_to_dtype_hold,safe_to_dtype_hold,load_file_mmap_to_dtype,load_file_pread_to_dtype,raw_read"))
    parser.add_argument("--raw-read-mib", type=int, default=int(os.environ.get("LOAD_BENCH_RAW_READ_MIB", "8")))
    parser.add_argument("--max-files", type=int, default=int(os.environ.get("LOAD_BENCH_MAX_FILES", "0")))
    return parser.parse_args()


def run_command(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True)
        text = (proc.stdout + proc.stderr).strip()
        return text.replace("\n", "\\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        return repr(exc)


def dtype_from_name(name: str):
    return {
        "none": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def faults() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_minflt, usage.ru_majflt


def log(message: str) -> None:
    print(f"[loadbench] {message}", flush=True)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def summarize_paths(paths: list[str]) -> tuple[int, int]:
    total = 0
    for path in paths:
        total += Path(path).stat().st_size
    return len(paths), total


def safe_open_scan(paths: list[str]) -> None:
    total_keys = 0
    dtype_counts: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            total_keys += len(keys)
            for key in keys:
                tensor = handle.get_tensor(key)
                dtype_name = str(tensor.dtype)
                dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
                byte_counts[dtype_name] = byte_counts.get(dtype_name, 0) + tensor_nbytes(tensor)
                del tensor
    log(f"TENSOR_SUMMARY keys={total_keys} dtype_counts={dtype_counts} byte_counts={byte_counts}")


def bench_raw_read(paths: list[str], chunk_mib: int, dtype) -> int:
    del dtype
    chunk_size = chunk_mib * 1024 * 1024
    total = 0
    checksum = 0
    for path in paths:
        with open(path, "rb", buffering=0) as handle:
            while True:
                data = handle.read(chunk_size)
                if not data:
                    break
                total += len(data)
                checksum ^= data[0]
    log(f"RAW_READ_CHECKSUM checksum={checksum}")
    return total


def bench_safe_get(paths: list[str], dtype) -> int:
    del dtype
    total = 0
    tensor_count = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                total += tensor_nbytes(tensor)
                tensor_count += 1
                del tensor
    log(f"SAFE_GET tensors={tensor_count}")
    return total


def bench_safe_to_dtype_hold(paths: list[str], dtype) -> int:
    state_dict = {}
    total = 0
    tensor_count = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if dtype is not None:
                    tensor = tensor.to(dtype)
                state_dict[key] = tensor
                total += tensor_nbytes(tensor)
                tensor_count += 1
    log(f"SAFE_TO_DTYPE_HOLD tensors={tensor_count} held_keys={len(state_dict)}")
    del state_dict
    return total


def load_file_one(path: str, backend: str | None):
    if backend is None:
        return load_file(path, device="cpu")
    try:
        signature = inspect.signature(load_file)
        if "backend" in signature.parameters:
            return load_file(path, device="cpu", backend=backend)
    except (TypeError, ValueError):
        pass
    return load_file(path, device="cpu")


def bench_load_file(paths: list[str], dtype, backend: str | None) -> int:
    state_dict = {}
    total = 0
    tensor_count = 0
    for path in paths:
        partial = load_file_one(path, backend=backend)
        for key, tensor in partial.items():
            if dtype is not None:
                tensor = tensor.to(dtype)
            state_dict[key] = tensor
            total += tensor_nbytes(tensor)
            tensor_count += 1
        del partial
    log(f"LOAD_FILE backend={backend or 'default'} tensors={tensor_count} held_keys={len(state_dict)}")
    del state_dict
    return total


def run_bench(name: str, func: Callable[[list[str], torch.dtype | None], int], paths: list[str], dtype) -> None:
    gc.collect()
    start_minflt, start_majflt = faults()
    start = time.perf_counter()
    bytes_processed = func(paths, dtype)
    elapsed = time.perf_counter() - start
    end_minflt, end_majflt = faults()
    mib = bytes_processed / (1024 * 1024)
    rate = mib / elapsed if elapsed > 0 else 0.0
    log(
        "BENCH_RESULT "
        f"method={name} seconds={elapsed:.3f} processed_gib={bytes_processed / (1024**3):.3f} "
        f"mib_per_s={rate:.2f} maxrss_gib={rss_gib():.2f} "
        f"minor_faults={end_minflt - start_minflt} major_faults={end_majflt - start_majflt}"
    )
    gc.collect()


def main() -> None:
    args = parse_args()
    root = Path(args.checkpoint_root)
    paths = sorted(glob.glob(str(root / args.pattern)))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No safetensors matched: {root / args.pattern}")

    dtype = dtype_from_name(args.dtype)
    file_count, total_size = summarize_paths(paths)
    log(
        "START "
        f"host={socket.gethostname()} python={sys.executable} torch={torch.__version__} "
        f"safetensors_root={root} pattern={args.pattern} files={file_count} "
        f"total_gib={total_size / (1024**3):.3f} dtype={dtype} "
        f"methods={args.methods}"
    )
    log("DF " + run_command(["df", "-h", "/", "/tmp", "/dev/shm", str(root)]))
    for path in paths:
        log(f"FILE path={path} size_gib={Path(path).stat().st_size / (1024**3):.3f}")

    run_bench("safe_open_scan", lambda p, d: (safe_open_scan(p) or total_size), paths, dtype)

    method_map: dict[str, Callable[[list[str], torch.dtype | None], int]] = {
        "raw_read": lambda p, d: bench_raw_read(p, args.raw_read_mib, d),
        "safe_get": bench_safe_get,
        "safe_to_dtype_hold": bench_safe_to_dtype_hold,
        "load_file_default_to_dtype": lambda p, d: bench_load_file(p, d, None),
        "load_file_mmap_to_dtype": lambda p, d: bench_load_file(p, d, "mmap"),
        "load_file_pread_to_dtype": lambda p, d: bench_load_file(p, d, "pread"),
    }

    for index, method in enumerate([item.strip() for item in args.methods.split(",") if item.strip()], start=1):
        func = method_map.get(method)
        if func is None:
            raise ValueError(f"Unknown benchmark method: {method}. Options: {sorted(method_map)}")
        run_bench(f"{index}_{method}", func, paths, dtype)

    log("DONE")


if __name__ == "__main__":
    main()
