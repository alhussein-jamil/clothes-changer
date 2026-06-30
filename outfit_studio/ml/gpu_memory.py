"""CUDA memory helpers — device selection from live VRAM, not fixed thresholds."""

from __future__ import annotations

import gc
import logging
import os
import threading
from contextlib import contextmanager

import torch

from outfit_studio.config import PROJECT_ROOT
from outfit_studio.constants import (
    BYTES_PER_GB,
    BYTES_PER_MIB,
    VRAM_INPAINT_CONTROLNET_GB,
    VRAM_INPAINT_PLAIN_GB,
    VRAM_INPAINT_SDXL_GB,
    VRAM_POSE_PEAK_GB,
    VRAM_SEGMENTATION_PEAK_GB,
)
from outfit_studio.ml.checkpoints import is_sdxl_model_name

logger = logging.getLogger(__name__)

_CONFIGURED = False
_MODEL_LOAD_LOCK = threading.Lock()


@contextmanager
def model_load_lock():
    """Serialize heavy checkpoint loads (diffusers + human parser share meta-device state)."""
    with _MODEL_LOAD_LOCK:
        yield


def configure_pytorch_memory() -> None:
    """Apply allocator and compile-cache settings before the first CUDA allocation."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    inductor_dir = os.environ.get("OUTFIT_STUDIO_INDUCTOR_CACHE_DIR")
    if not inductor_dir:
        inductor_dir = str(PROJECT_ROOT / ".cache" / "torchinductor")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    logger.info(
        "PyTorch CUDA allocator configured (expandable_segments=True, inductor_cache=%s)",
        inductor_dir,
    )


def free_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / BYTES_PER_MIB
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / BYTES_PER_MIB
        logger.debug("CUDA cache cleared (%.0f → %.0f MiB allocated)", before, after)


def gpu_memory_gb() -> tuple[float, float]:
    """Return (free_gb, total_gb) for the default CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return free_bytes / BYTES_PER_GB, total_bytes / BYTES_PER_GB


def gpu_total_gb() -> float:
    return gpu_memory_gb()[1]


def gpu_free_gb() -> float:
    return gpu_memory_gb()[0]


def _inpaint_vram_budget_gb() -> float:
    """Estimate peak VRAM inpainting needs so segmentation can yield the GPU."""
    from outfit_studio.config import get_settings

    settings = get_settings()
    if is_sdxl_model_name(settings.inpaint_model):
        return VRAM_INPAINT_SDXL_GB
    return VRAM_INPAINT_CONTROLNET_GB if settings.use_controlnet else VRAM_INPAINT_PLAIN_GB


def _combined_ml_vram_gb() -> float:
    """Peak VRAM when segmentation, pose ONNX, and inpaint may all be resident."""
    return VRAM_SEGMENTATION_PEAK_GB + VRAM_POSE_PEAK_GB + _inpaint_vram_budget_gb()


def both_stacks_fit_on_gpu() -> bool:
    """True when total VRAM can hold segmentation, pose, and inpaint together."""
    if not torch.cuda.is_available():
        return False
    _, total_gb = gpu_memory_gb()
    return total_gb >= _combined_ml_vram_gb()


def segmentation_uses_cuda() -> bool:
    return not prefer_cpu_for_segmentation()


def prepare_for_segmentation() -> None:
    """Free inpaint VRAM only when GPU segmentation cannot run alongside it."""
    if not segmentation_uses_cuda():
        return
    from outfit_studio.ml.inpainter import get_inpaint_engine

    engine = get_inpaint_engine()
    if not engine.is_loaded():
        return
    if both_stacks_fit_on_gpu() and gpu_free_gb() >= VRAM_SEGMENTATION_PEAK_GB:
        return
    release_inpaint_gpu()


def prepare_for_inpaint() -> None:
    """Free segmentation VRAM only when inpaint must load and space is tight."""
    if not segmentation_uses_cuda():
        return
    from outfit_studio.ml.inpainter import get_inpaint_engine
    from outfit_studio.ml.segmentor import get_segmentor

    engine = get_inpaint_engine()
    if engine.is_loaded():
        return
    segmentor = get_segmentor()
    if not segmentor.is_loaded():
        return
    if both_stacks_fit_on_gpu() and gpu_free_gb() >= _inpaint_vram_budget_gb():
        return
    release_segmentation_gpu()


def prefer_cpu_for_segmentation() -> bool:
    """Keep the human parser on CPU only when the card cannot fit seg + inpaint.

    Uses total VRAM capacity, not transient free memory (inpaint may be loaded
    while the user re-segments). Segmentation stays on GPU whenever the card
    has enough total memory for both stacks.
    """
    if not torch.cuda.is_available():
        logger.debug("Segmentation on CPU (no CUDA)")
        return True

    free_gb, total_gb = gpu_memory_gb()
    combined_gb = _combined_ml_vram_gb()
    use_cpu = total_gb < combined_gb

    logger.debug(
        "Segmentation device: %s (free=%.1f GB total=%.1f GB need=%.1f GB inpaint_budget=%.1f GB)",
        "CPU" if use_cpu else "CUDA",
        free_gb,
        total_gb,
        combined_gb,
        _inpaint_vram_budget_gb(),
    )
    return use_cpu


def release_segmentation_gpu() -> None:
    """Drop segmentation weights from VRAM (models reload on next use)."""
    logger.info("Releasing segmentation models from GPU")
    from outfit_studio.ml.segmentor import get_segmentor

    get_segmentor().unload()
    free_cuda_cache()


def release_inpaint_gpu() -> None:
    logger.info("Releasing inpaint pipeline from GPU")
    from outfit_studio.ml.inpainter import get_inpaint_engine

    get_inpaint_engine().unload()
    free_cuda_cache()
