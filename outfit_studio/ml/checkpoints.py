"""Checkpoint validation and architecture detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Architecture = Literal["sd15", "sdxl"]

SDXL_NAME_HINTS = ("sdxl", "xl_inpaint", "xl-inpaint", "xl_inpainting")
_SDXL_KEY_MARKERS = ("conditioner.", "text_encoder_2.")


def is_sdxl_model_name(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in SDXL_NAME_HINTS)


def _safetensors_keys(path: Path) -> list[str] | None:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return list(handle.keys())
    except Exception as exc:
        logger.debug("Could not read safetensors keys from %s: %s", path.name, exc)
        return None


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


def inpaint_checkpoint_valid(path: Path) -> bool:
    """Return whether *path* is a complete, loadable inpaint checkpoint."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return _safetensors_readable(path)
    if suffix == ".ckpt":
        return _pickle_checkpoint_readable(path)
    return False


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
    return "sdxl" if is_sdxl_checkpoint(name, path) else "sd15"
