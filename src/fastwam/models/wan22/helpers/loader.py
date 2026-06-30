from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import inspect
import os
from typing import Any

import torch
import time

from .io import ModelConfig, _debug_event, _debug_path_summary, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


@contextmanager
def _maybe_serialize_local_load(model_name: str):
    if not _env_flag("FASTWAM_SERIALIZE_WAN_LOAD"):
        yield
        return

    lock_dir = os.environ.get("FASTWAM_LOAD_LOCK_DIR", "/tmp/fastwam_load_locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{model_name}.lock")
    start = time.time()
    _debug_event("load_lock_wait_start", model_name=model_name, lock_path=lock_path)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _debug_event("load_lock_acquired", model_name=model_name, seconds=f"{time.time() - start:.2f}")
        try:
            yield
        finally:
            _debug_event("load_lock_release", model_name=model_name)
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    total_start = time.time()
    _debug_event("load_registered_start", model_name=model_name, path=_debug_path_summary(path), device=device, dtype=torch_dtype)
    with _maybe_serialize_local_load(model_name):
        start = time.time()
        model_hash = hash_model_file(path)
        _debug_event("load_registered_hash_done", model_name=model_name, model_hash=model_hash, seconds=f"{time.time() - start:.2f}")

        matched_config = None
        for config in WAN22_MODEL_REGISTRY:
            if config["model_hash"] == model_hash and config["model_name"] == model_name:
                matched_config = config
                break
        if matched_config is None:
            raise ValueError(
                f"Cannot detect model type for {model_name}. File: {path}. "
                f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
            )

        model_class = matched_config["model_class"]
        model_kwargs = dict(matched_config.get("extra_kwargs", {}))
        if model_kwargs_override is not None:
            model_kwargs.update(model_kwargs_override)
        state_dict_converter = matched_config.get("state_dict_converter")

        start = time.time()
        _debug_event("model_init_start", model_name=model_name, model_class=model_class.__name__)
        model = model_class(**model_kwargs)
        _debug_event("model_init_done", model_name=model_name, seconds=f"{time.time() - start:.2f}")

        start = time.time()
        state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
        _debug_event("state_dict_load_done", model_name=model_name, keys=len(state_dict), seconds=f"{time.time() - start:.2f}")
        if state_dict_converter is not None:
            start = time.time()
            _debug_event("state_dict_convert_start", model_name=model_name)
            state_dict = state_dict_converter(state_dict)
            _debug_event("state_dict_convert_done", model_name=model_name, keys=len(state_dict), seconds=f"{time.time() - start:.2f}")

        start = time.time()
        _debug_event("load_state_dict_into_model_start", model_name=model_name)
        model.load_state_dict(state_dict, strict=False)
        _debug_event("load_state_dict_into_model_done", model_name=model_name, seconds=f"{time.time() - start:.2f}")
        start = time.time()
        _debug_event("model_to_device_start", model_name=model_name, device=device, dtype=torch_dtype)
        model = model.to(device=device, dtype=torch_dtype)
        _debug_event("model_to_device_done", model_name=model_name, seconds=f"{time.time() - start:.2f}")
    _debug_event("load_registered_done", model_name=model_name, seconds=f"{time.time() - total_start:.2f}")
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()
    _debug_event(
        "wan22_load_start",
        device=device,
        dtype=torch_dtype,
        base_path=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"),
        skip_download=os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD"),
        load_text_encoder=load_text_encoder,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    _debug_event("vae_download_if_necessary_start")
    vae_config.download_if_necessary()
    _debug_event("vae_download_if_necessary_done", path=_debug_path_summary(vae_config.path))
    if load_text_encoder:
        _debug_event("text_download_if_necessary_start")
        text_config.download_if_necessary()
        _debug_event("text_download_if_necessary_done", path=_debug_path_summary(text_config.path))
        _debug_event("tokenizer_download_if_necessary_start")
        tokenizer_config.download_if_necessary()
        _debug_event("tokenizer_download_if_necessary_done", path=_debug_path_summary(tokenizer_config.path))

    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        _debug_event("dit_download_if_necessary_start")
        dit_model_config.download_if_necessary()
        _debug_event("dit_download_if_necessary_done", path=_debug_path_summary(dit_model_config.path))
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        _debug_event("text_encoder_load_start", path=_debug_path_summary(text_config.path))
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
        _debug_event("text_encoder_load_done", path=text_encoder_path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    _debug_event("vae_load_start", path=_debug_path_summary(vae_config.path))
    vae: WanVideoVAE38 = _load_registered_model(vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device)
    _debug_event("vae_load_done", path=str(vae_config.path))
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    _debug_event("wan22_load_done", seconds=f"{time.time() - start:.2f}")
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
