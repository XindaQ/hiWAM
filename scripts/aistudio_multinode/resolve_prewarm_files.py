#!/usr/bin/env python3
"""Resolve the checkpoint files that Wan2.2 training will read.

This intentionally follows the same Hydra config and Wan loader config rules as
training, then prints one absolute file path per line for prewarming.
Diagnostics go to stderr so the shell script can consume stdout safely.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastwam.models.wan22.helpers.loader import _resolve_configs
from fastwam.utils.config_resolvers import register_default_resolvers


def eprint(message: str) -> None:
    print(f"[prewarm_resolve] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def compose_training_cfg(project_root: Path, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    config_dir = str(project_root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def expand_patterns(root: Path, model_id: str, origin_file_pattern: str | list[str], label: str) -> list[Path]:
    patterns = origin_file_pattern if isinstance(origin_file_pattern, list) else [origin_file_pattern]
    paths: list[Path] = []
    for pattern in patterns:
        pattern = "*" if pattern in (None, "", "./") else str(pattern)
        if pattern.endswith("/"):
            pattern = pattern + "*"
        full_pattern = root / model_id / pattern
        matches = [Path(item) for item in glob.glob(str(full_pattern))]
        files = sorted(path for path in matches if path.is_file() or path.is_symlink())
        eprint(f"{label}: pattern={full_pattern} files={len(files)}")
        paths.extend(files)
    return paths


def action_dit_path(project_root: Path, model_cfg: DictConfig) -> Path | None:
    if bool(model_cfg.get("skip_dit_load_from_pretrain", False)):
        return None
    value = model_cfg.get("action_dit_pretrained_path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path


def unique_existing(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    seen: set[str] = set()
    existing: list[Path] = []
    missing: list[Path] = []
    for path in paths:
        path = path.resolve() if path.exists() else path
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() or path.is_symlink():
            existing.append(path)
        else:
            missing.append(path)
    return existing, missing


def main() -> int:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    project_root = Path(args.project_root)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(checkpoint_root)

    eprint(f"checkpoint_root={checkpoint_root}")
    eprint(f"project_root={project_root}")
    eprint(f"overrides={args.overrides}")

    cfg = compose_training_cfg(project_root, args.overrides)
    model_cfg = cfg.model

    model_id = str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    load_text_encoder = bool(model_cfg.get("load_text_encoder", True))
    skip_dit = bool(model_cfg.get("skip_dit_load_from_pretrain", False))

    eprint(
        "model "
        f"model_id={model_id} tokenizer_model_id={tokenizer_model_id} "
        f"redirect_common_files={redirect_common_files} "
        f"load_text_encoder={load_text_encoder} skip_dit_load_from_pretrain={skip_dit}"
    )

    dit_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    paths: list[Path] = []
    if not skip_dit:
        paths.extend(
            expand_patterns(
                checkpoint_root,
                str(dit_config.model_id),
                dit_config.parse_original_file_pattern(),
                "dit",
            )
        )
    paths.extend(
        expand_patterns(
            checkpoint_root,
            str(vae_config.model_id),
            vae_config.parse_original_file_pattern(),
            "vae",
        )
    )

    if load_text_encoder:
        paths.extend(
            expand_patterns(
                checkpoint_root,
                str(text_config.model_id),
                text_config.parse_original_file_pattern(),
                "text",
            )
        )
        paths.extend(
            expand_patterns(
                checkpoint_root,
                str(tokenizer_config.model_id),
                tokenizer_config.parse_original_file_pattern(),
                "tokenizer",
            )
        )

    action_path = action_dit_path(project_root, model_cfg)
    if action_path is not None:
        eprint(f"action_dit path={action_path}")
        paths.append(action_path)

    existing, missing = unique_existing(paths)
    if missing:
        for path in missing:
            eprint(f"missing={path}")
        return 2

    if not existing:
        eprint("no checkpoint files resolved")
        return 2

    for path in existing:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
