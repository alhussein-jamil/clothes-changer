"""Checkpoint validation and architecture detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Architecture = Literal["sd15", "sdxl"]

SDXL_NAME_HINTS = ("sdxl", "xl_inpaint", "xl-inpaint", "xl_inpainting")
_SDXL_KEY_MARKERS = ("conditioner.", "text_encoder_2.")

_SAFETENSORS_KEYS_CACHE: dict[tuple[str, int, int], list[str] | None] = {}
_VALIDATION_CACHE: dict[tuple[str, int, int], bool] = {}
_ARCHITECTURE_CACHE: dict[tuple[str, int, int], Architecture] = {}


def clear_checkpoint_cache() -> None:
    """Drop cached safetensors metadata (call after downloads or model dir changes)."""
    _SAFETENSORS_KEYS_CACHE.clear()
    _VALIDATION_CACHE.clear()
    _ARCHITECTURE_CACHE.clear()


def _file_cache_key(path: Path) -> tuple[str, int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    if stat.st_size == 0:
        return None
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def is_sdxl_model_name(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in SDXL_NAME_HINTS)


def _safetensors_keys(path: Path) -> list[str] | None:
    cache_key = _file_cache_key(path)
    if cache_key is not None and cache_key in _SAFETENSORS_KEYS_CACHE:
        return _SAFETENSORS_KEYS_CACHE[cache_key]

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        logger.debug("Could not read safetensors keys from %s: %s", path.name, exc)
        keys = None

    if cache_key is not None:
        _SAFETENSORS_KEYS_CACHE[cache_key] = keys
    return keys


def _safetensors_readable(path: Path) -> bool:
    keys = _safetensors_keys(path)
    if not keys:
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            handle.get_tensor(keys[0])
        return True
    except Exception as exc:
        logger.debug("Safetensors tensor read failed for %s: %s", path.name, exc)
        return False


def _pickle_checkpoint_readable(path: Path) -> bool:
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(state, dict) and bool(state)
    except Exception as exc:
        logger.debug("Checkpoint load failed for %s: %s", path.name, exc)
        return False


def inpaint_checkpoint_listable(path: Path) -> bool:
    """Fast validation for model discovery (header/keys only, no tensor read)."""
    cache_key = _file_cache_key(path)
    if cache_key is None:
        return False
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return bool(_safetensors_keys(path))
    if suffix == ".ckpt":
        return path.stat().st_size > 1024
    return False


def inpaint_checkpoint_valid(path: Path) -> bool:
    """Return whether *path* is a complete, loadable inpaint checkpoint."""
    cache_key = _file_cache_key(path)
    if cache_key is None:
        return False
    if cache_key in _VALIDATION_CACHE:
        return _VALIDATION_CACHE[cache_key]

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        valid = _safetensors_readable(path)
    elif suffix == ".ckpt":
        valid = _pickle_checkpoint_readable(path)
    else:
        valid = False

    _VALIDATION_CACHE[cache_key] = valid
    return valid


def is_hub_model_id(model_id: str) -> bool:
    """Return True for Hugging Face repo ids (e.g. ``org/model``)."""
    if model_id.endswith((".safetensors", ".ckpt")):
        return False
    return "/" in model_id


def is_sdxl_checkpoint(name: str, path: Path) -> bool:
    if is_sdxl_model_name(name):
        return True
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    keys = _safetensors_keys(path)
    if not keys:
        return False
    return any(marker in key for key in keys for marker in _SDXL_KEY_MARKERS)


def checkpoint_architecture(name: str, path: Path) -> Architecture:
    cache_key = _file_cache_key(path)
    if cache_key is not None and cache_key in _ARCHITECTURE_CACHE:
        return _ARCHITECTURE_CACHE[cache_key]

    arch: Architecture = "sdxl" if is_sdxl_checkpoint(name, path) else "sd15"
    if cache_key is not None:
        _ARCHITECTURE_CACHE[cache_key] = arch
    return arch
