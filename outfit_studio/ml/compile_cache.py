"""Persist torch.compile artifacts between application restarts."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def cache_key(model_id: str, arch: str, use_controlnet: bool) -> str:
    """Stable filename stem keyed by torch version, model, and pipeline options."""
    # v2: compile UNet only (ControlNet left eager — avoids cudagraph crashes).
    raw = f"v2|torch={torch.__version__}|model={model_id}|arch={arch}|cn={int(use_controlnet)}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(model_id).name)[:48]
    return f"{slug}_{digest}"


def artifact_path(cache_dir: Path, model_id: str, arch: str, use_controlnet: bool) -> Path:
    return cache_dir / f"{cache_key(model_id, arch, use_controlnet)}.ptc"


def load_artifacts(cache_dir: Path, model_id: str, arch: str, use_controlnet: bool) -> bool:
    """Restore a previously saved torch.compile cache. Returns True when loaded."""
    path = artifact_path(cache_dir, model_id, arch, use_controlnet)
    if not path.is_file():
        logger.debug("No torch.compile cache at %s", path)
        return False
    try:
        info = torch.compiler.load_cache_artifacts(path.read_bytes())
        if info is None:
            logger.warning("torch.compile cache at %s could not be applied", path)
            return False
        logger.info("Loaded torch.compile cache from %s", path)
        return True
    except Exception as exc:
        logger.warning("Failed to load torch.compile cache %s: %s", path, exc)
        return False


def save_artifacts(cache_dir: Path, model_id: str, arch: str, use_controlnet: bool) -> bool:
    """Serialize torch.compile artifacts produced during warmup/inference."""
    try:
        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            logger.debug("No torch.compile artifacts to save yet")
            return False
        artifact_bytes, cache_info = saved
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_path(cache_dir, model_id, arch, use_controlnet)
        path.write_bytes(artifact_bytes)
        logger.info(
            "Saved torch.compile cache → %s (%d bytes, %s)",
            path,
            len(artifact_bytes),
            cache_info,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to save torch.compile cache: %s", exc)
        return False
