#!/usr/bin/env python3
"""Copy only the resolved checkpoint files behind local symlinks.

Input is a file list in the local checkpoint tree, produced from the same Hydra
config as training.

For each listed path:
  dst = local path training will read
  src = dst.resolve()

We first record all real sources, then replace symlink directories with real
local directories and copy just those files. No recursive dereference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import time


def log(message: str) -> None:
    print(f"[materialize] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--chunk-mb", type=int, default=64)
    return parser.parse_args()


def first_symlink_parent(path: Path, root: Path) -> Path | None:
    current = root
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return path if path.is_symlink() else None

    for part in parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def copy_file(src: Path, dst: Path, chunk_size: int) -> None:
    size = src.stat().st_size
    tmp = dst.with_name(f".{dst.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    copied = 0
    started = time.perf_counter()
    last_log = started

    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"copy_begin src={src} dst={dst} size_mib={size // (1024 * 1024)}")
    with src.open("rb", buffering=0) as reader, tmp.open("wb", buffering=0) as writer:
        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            writer.write(chunk)
            copied += len(chunk)
            now = time.perf_counter()
            if now - last_log >= 10:
                elapsed = max(now - started, 0.001)
                log(
                    f"progress dst={dst} copied_mib={copied // (1024 * 1024)} "
                    f"total_mib={size // (1024 * 1024)} "
                    f"mib_per_s={copied / (1024 * 1024) / elapsed:.2f}"
                )
                last_log = now

    os.replace(tmp, dst)
    elapsed = max(time.perf_counter() - started, 0.001)
    log(
        f"copy_done dst={dst} seconds={elapsed:.3f} "
        f"size_mib={size // (1024 * 1024)} "
        f"mib_per_s={size / (1024 * 1024) / elapsed:.2f}"
    )


def mib(size: int) -> int:
    return size // (1024 * 1024)


def log_disk(path: Path, label: str) -> None:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        log(f"disk label={label} path={path} error={exc}")
        return
    log(
        f"disk label={label} path={path} "
        f"total_gib={usage.total // (1024 ** 3)} "
        f"used_gib={usage.used // (1024 ** 3)} "
        f"free_gib={usage.free // (1024 ** 3)}"
    )


def main() -> int:
    args = parse_args()
    root = Path(args.checkpoint_root)
    file_list = Path(args.file_list)
    chunk_size = args.chunk_mb * 1024 * 1024

    paths = [Path(line.strip()) for line in file_list.read_text().splitlines() if line.strip()]
    log(f"host={socket.gethostname()}")
    log(f"root={root}")
    log(f"file_list={file_list}")
    log(f"chunk_mb={args.chunk_mb}")
    log(f"file_count={len(paths)}")
    log_disk(root, "before")

    copies: list[tuple[Path, Path, Path]] = []
    copy_total_bytes = 0
    for dst in paths:
        if not dst.exists():
            log(f"missing path={dst}")
            return 2
        link = first_symlink_parent(dst, root)
        if link is None:
            log(f"real_file path={dst}")
            continue
        src = dst.resolve()
        size = src.stat().st_size
        copy_total_bytes += size
        log(
            f"inspect dst={dst} symlink_parent={link} "
            f"src={src} size_mib={mib(size)}"
        )
        copies.append((src, dst, link))

    log(f"copy_count={len(copies)} copy_total_mib={mib(copy_total_bytes)}")
    for link in sorted({item[2] for item in copies}, key=lambda path: len(path.parts), reverse=True):
        if link.is_symlink():
            log(f"replace_symlink path={link} target={link.readlink()}")
            link.unlink()
            link.mkdir(parents=True, exist_ok=True)
            log(f"replace_done path={link} is_symlink={link.is_symlink()}")

    started = time.perf_counter()
    for index, (src, dst, _) in enumerate(copies, start=1):
        log(f"copying index={index}/{len(copies)}")
        copy_file(src, dst, chunk_size)

    elapsed = max(time.perf_counter() - started, 0.001)
    log_disk(root, "after")
    log(f"done seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
