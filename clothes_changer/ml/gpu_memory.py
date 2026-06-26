"""CUDA memory helpers — device selection from live VRAM, not fixed thresholds."""

from __future__ import annotations

import gc
import logging
import os

import torch

from clothes_changer.ml.checkpoints import is_sdxl_model_name

logger = logging.getLogger(__name__)

_CONFIGURED = False

# Peak VRAM while segmentation models are loaded (SegFormer-b2 + U2NET).
_SEGMENTATION_PEAK_GB = 2.0


def configure_pytorch_memory() -> None:
    """Apply allocator settings before the first CUDA allocation."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info("PyTorch CUDA allocator configured (expandable_segments=True)")


def free_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / (1024**2)
        logger.debug("CUDA cache cleared (%.0f → %.0f MiB allocated)", before, after)


def gpu_memory_gb() -> tuple[float, float]:
    """Return (free_gb, total_gb) for the default CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gb = 1024**3
    return free_bytes / gb, total_bytes / gb


def gpu_total_gb() -> float:
    return gpu_memory_gb()[1]


def gpu_free_gb() -> float:
    return gpu_memory_gb()[0]


def _inpaint_vram_budget_gb() -> float:
    """Estimate peak VRAM inpainting needs so segmentation can yield the GPU."""
    from clothes_changer.config import get_settings

    settings = get_settings()
    if is_sdxl_model_name(settings.inpaint_model):
        return 10.0
    return 6.0 if settings.use_controlnet else 4.5


def prefer_cpu_for_segmentation() -> bool:
    """Keep SegFormer/U2NET on CPU only when the card cannot fit seg + inpaint.

    Uses total VRAM capacity, not transient free memory (inpaint may be loaded
    while the user re-segments). Segmentation stays on GPU whenever the card
    has enough total memory for both stacks.
    """
    if not torch.cuda.is_available():
        logger.debug("Segmentation on CPU (no CUDA)")
        return True

    free_gb, total_gb = gpu_memory_gb()
    inpaint_gb = _inpaint_vram_budget_gb()
    combined_gb = _SEGMENTATION_PEAK_GB + inpaint_gb
    use_cpu = total_gb < combined_gb

    logger.debug(
        "Segmentation device: %s (free=%.1f GB total=%.1f GB need=%.1f GB inpaint_budget=%.1f GB)",
        "CPU" if use_cpu else "CUDA",
        free_gb,
        total_gb,
        combined_gb,
        inpaint_gb,
    )
    return use_cpu


def release_segmentation_gpu() -> None:
    """Drop segmentation weights from VRAM (models reload on next use)."""
    logger.info("Releasing segmentation models from GPU")
    from clothes_changer.ml.segmentor import get_segmentor

    get_segmentor().unload()
    free_cuda_cache()


def release_inpaint_gpu() -> None:
    logger.info("Releasing inpaint pipeline from GPU")
    from clothes_changer.ml.inpainter import get_inpaint_engine

    get_inpaint_engine().unload()
    free_cuda_cache()
