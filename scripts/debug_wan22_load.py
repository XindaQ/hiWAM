#!/usr/bin/env python3
"""Profile Wan2.2 / FastWAM model loading stages."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.helpers.io import hash_model_file, load_state_dict
from fastwam.models.wan22.helpers.loader import (
    WAN22_MODEL_REGISTRY,
    _resolve_configs,
    _validate_dit_config,
)
from fastwam.utils.config_resolvers import register_default_resolvers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--mode", choices=["detailed", "full"], default="detailed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def log(message: str) -> None:
    rank, local_rank, _ = rank_info()
    print(f"[load-debug][rank={rank} local_rank={local_rank}] {message}", flush=True)


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start = time.perf_counter()
        log(f"BEGIN {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - self.start
        log(f"END {self.name} seconds={elapsed:.3f} mem={memory_summary()}")


def memory_summary() -> str:
    rss = "unknown"
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = " ".join(line.split()[1:3])
                    break
    except OSError:
        pass
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"rss={rss} cuda_alloc={allocated:.2f}GiB cuda_reserved={reserved:.2f}GiB"
    return f"rss={rss}"


def run_command(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return repr(exc)


def path_size(path: Any) -> str:
    if isinstance(path, list):
        total = 0
        for item in path:
            try:
                total += Path(item).stat().st_size
            except OSError:
                pass
        return f"{len(path)} files total={total / (1024**3):.2f}GiB"
    try:
        return f"{Path(path).stat().st_size / (1024**3):.2f}GiB"
    except OSError:
        return "missing"


def compose_cfg(task: str, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}", *overrides])
    OmegaConf.resolve(cfg)
    return cfg


def dtype_from_arg(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def device_from_arg(name: str) -> str:
    _, local_rank, _ = rank_info()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return name


def find_registry_config(model_hash: str, model_name: str) -> dict[str, Any]:
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            return config
    raise RuntimeError(f"No registry match for {model_name} hash={model_hash}")


def load_registered_model_profiled(path, model_name: str, torch_dtype: torch.dtype, device: str, model_kwargs=None):
    model_kwargs = dict(model_kwargs or {})
    with Timer(f"{model_name}:hash_model_file"):
        model_hash = hash_model_file(path)
    log(f"{model_name}:hash={model_hash} path={path} size={path_size(path)}")

    registry_config = find_registry_config(model_hash, model_name)
    model_class = registry_config["model_class"]
    kwargs = dict(registry_config.get("extra_kwargs", {}))
    kwargs.update(model_kwargs)
    state_dict_converter = registry_config.get("state_dict_converter")

    with Timer(f"{model_name}:construct_module"):
        model = model_class(**kwargs)

    with Timer(f"{model_name}:load_state_dict_file_cpu"):
        state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    log(f"{model_name}:state_dict_keys={len(state_dict)}")

    if state_dict_converter is not None:
        with Timer(f"{model_name}:state_dict_converter"):
            state_dict = state_dict_converter(state_dict)

    with Timer(f"{model_name}:module_load_state_dict"):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log(f"{model_name}:missing={len(missing)} unexpected={len(unexpected)}")

    del state_dict

    with Timer(f"{model_name}:to_device_dtype"):
        model = model.to(device=device, dtype=torch_dtype)
    return model


def resolve_model_paths(model_cfg: DictConfig):
    with Timer("resolve_wan22_configs"):
        dit_config, text_config, vae_config, tokenizer_config = _resolve_configs(
            model_id=str(model_cfg.model_id),
            tokenizer_model_id=str(model_cfg.tokenizer_model_id),
            redirect_common_files=bool(model_cfg.redirect_common_files),
        )

    with Timer("vae_config.download_if_necessary"):
        vae_config.download_if_necessary()
    with Timer("dit_config.download_if_necessary"):
        dit_config.download_if_necessary()
    if bool(model_cfg.load_text_encoder):
        with Timer("text_config.download_if_necessary"):
            text_config.download_if_necessary()
        with Timer("tokenizer_config.download_if_necessary"):
            tokenizer_config.download_if_necessary()

    log(f"dit_path={dit_config.path} size={path_size(dit_config.path)}")
    log(f"vae_path={vae_config.path} size={path_size(vae_config.path)}")
    if bool(model_cfg.load_text_encoder):
        log(f"text_path={text_config.path} size={path_size(text_config.path)}")
        log(f"tokenizer_path={tokenizer_config.path}")
    return dit_config, vae_config


def detailed_profile(cfg: DictConfig, torch_dtype: torch.dtype, device: str) -> None:
    model_cfg = cfg.model
    video_dit_config = OmegaConf.to_container(model_cfg.video_dit_config, resolve=True)
    action_dit_config = OmegaConf.to_container(model_cfg.action_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise TypeError("video_dit_config did not resolve to dict")
    if not isinstance(action_dit_config, dict):
        raise TypeError("action_dit_config did not resolve to dict")

    with Timer("validate_video_dit_config"):
        video_dit_config = _validate_dit_config(video_dit_config)

    dit_config, vae_config = resolve_model_paths(model_cfg)

    video_expert = load_registered_model_profiled(
        dit_config.path,
        "wan_video_dit",
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=video_dit_config,
    )

    vae = load_registered_model_profiled(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    )

    with Timer("action_dit_from_pretrained"):
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=str(model_cfg.action_dit_pretrained_path),
            skip_dit_load_from_pretrain=bool(model_cfg.skip_dit_load_from_pretrain),
            device=device,
            torch_dtype=torch_dtype,
        )

    log(
        "loaded_modules "
        f"video_params={sum(p.numel() for p in video_expert.parameters())} "
        f"vae_params={sum(p.numel() for p in vae.parameters())} "
        f"action_params={sum(p.numel() for p in action_expert.parameters())}"
    )


def full_profile(cfg: DictConfig, torch_dtype: torch.dtype, device: str) -> None:
    with Timer("hydra_instantiate_cfg_model_full"):
        model = instantiate(cfg.model, model_dtype=torch_dtype, device=device)
    log(f"full_model_class={type(model).__name__}")
    log(f"full_model_paths={getattr(model, 'model_paths', None)}")


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = rank_info()
    device = device_from_arg(args.device)
    torch_dtype = dtype_from_arg(args.dtype)

    log(
        "START "
        f"host={socket.gethostname()} rank={rank} local_rank={local_rank} world_size={world_size} "
        f"cwd={os.getcwd()} python={sys.executable} torch={torch.__version__} "
        f"cuda={torch.version.cuda} device={device} dtype={torch_dtype} "
        f"DIFFSYNTH_MODEL_BASE_PATH={os.environ.get('DIFFSYNTH_MODEL_BASE_PATH')}"
    )
    log("df=" + run_command(["df", "-h", "/tmp", "/ossfs", "/team"]))

    with Timer("hydra_compose_config"):
        cfg = compose_cfg(args.task, args.overrides)

    if args.mode == "detailed":
        detailed_profile(cfg, torch_dtype=torch_dtype, device=device)
    else:
        full_profile(cfg, torch_dtype=torch_dtype, device=device)

    log("DONE")


if __name__ == "__main__":
    main()
