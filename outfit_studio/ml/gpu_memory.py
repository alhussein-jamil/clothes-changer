"""CUDA memory helpers — device selection from live VRAM, not fixed thresholds."""

from __future__ import annotations

import gc
import logging
import os
import threading
from contextlib import contextmanager

import torch

from outfit_studio.constants import (
    BYTES_PER_GB,
    BYTES_PER_MIB,
    VRAM_INPAINT_CONTROLNET_GB,
    VRAM_INPAINT_PLAIN_GB,
    VRAM_INPAINT_SDXL_GB,
    VRAM_POSE_PEAK_GB,
    VRAM_SEGMENTATION_PEAK_GB,
    VRAM_TIGHT_TOTAL_GB,
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
    from outfit_studio.config import get_settings

    settings = get_settings()
    inductor_dir = str(settings.resolved_inductor_cache_dir)
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


def vram_is_tight() -> bool:
    """True on consumer cards that cannot hold pose + seg + ControlNet inpaint."""
    if not torch.cuda.is_available():
        return False
    return gpu_total_gb() < VRAM_TIGHT_TOTAL_GB


def _inpaint_vram_budget_gb() -> float:
    """Estimate peak VRAM inpainting needs so segmentation can yield the GPU."""
    from outfit_studio.config import get_settings

    settings = get_settings()
    if is_sdxl_model_name(settings.content.default_inpaint):
        return VRAM_INPAINT_SDXL_GB
    return VRAM_INPAINT_CONTROLNET_GB if settings.content.use_controlnet else VRAM_INPAINT_PLAIN_GB


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


def prepare_for_pose() -> None:
    """Free PyTorch GPU residents so ONNX Runtime CUDA arenas can allocate.

    Pose runs *before* inpaint load in the generation pipeline, but background
    preload / a prior generate may still hold the UNet. ORT uses its own BFC
    arena and fails hard when PyTorch has the card filled.
    """
    if not torch.cuda.is_available():
        return

    from outfit_studio.ml.pose import get_pose_estimator

    # Already warm + enough free → skip empty_cache thrash and unload checks.
    if get_pose_estimator().sessions_loaded and gpu_free_gb() >= VRAM_POSE_PEAK_GB:
        return

    free_cuda_cache()
    if gpu_free_gb() >= VRAM_POSE_PEAK_GB:
        return

    from outfit_studio.ml.inpainter import get_inpaint_engine

    if get_inpaint_engine().is_loaded():
        logger.info(
            "Freeing inpaint GPU for pose (free=%.2f GB < need=%.2f GB)",
            gpu_free_gb(),
            VRAM_POSE_PEAK_GB,
        )
        release_inpaint_gpu()
        free_cuda_cache()

    if gpu_free_gb() >= VRAM_POSE_PEAK_GB:
        return

    from outfit_studio.ml.segmentor import get_segmentor

    if get_segmentor().is_loaded() and segmentation_uses_cuda():
        logger.info(
            "Freeing segmentation GPU for pose (free=%.2f GB < need=%.2f GB)",
            gpu_free_gb(),
            VRAM_POSE_PEAK_GB,
        )
        release_segmentation_gpu()
        free_cuda_cache()


def prepare_next_generate(*, use_controlnet: bool) -> None:
    """Warm pose for the next ControlNet run when VRAM allows keeping inpaint loaded."""
    if not use_controlnet or not torch.cuda.is_available():
        return
    if gpu_free_gb() < VRAM_POSE_PEAK_GB:
        return
    from outfit_studio.ml.pose import ensure_pose_on_gpu, get_pose_estimator

    if get_pose_estimator().sessions_loaded and get_pose_estimator().device == "cuda":
        return
    try:
        ensure_pose_on_gpu()
        logger.info("Next-run prep: pose warm (inpaint kept)")
    except Exception as exc:
        logger.debug("Next-run pose prep skipped (%s)", exc)


def release_pose_gpu() -> None:
    """Drop rtmlib ONNX CUDA sessions (safe once bboxes/keypoints are done)."""
    from outfit_studio.ml.pose import get_pose_estimator

    est = get_pose_estimator()
    if est.device == "cuda" and (est._det is not None or est._pose is not None):
        logger.info("Releasing pose/detector ONNX from GPU")
    est.unload()
    free_cuda_cache()


def prepare_for_inpaint() -> None:
    """Yield GPU to the inpaint pipeline (pose + segmentation must not linger).

    On 8 GB cards, SD1.5 inpaint + ControlNet needs most of the device. Pose
    ONNX and the human parser have already produced their outputs by the time
    this runs (or can reload on CPU), so they must leave VRAM first.
    """
    release_pose_gpu()

    from outfit_studio.ml.segmentor import get_segmentor

    segmentor = get_segmentor()
    if segmentor.is_loaded() and (
        segmentation_uses_cuda() or gpu_free_gb() < _inpaint_vram_budget_gb()
    ):
        # Free CUDA-resident parser; also drop when VRAM is below the inpaint budget.
        release_segmentation_gpu()

    free_cuda_cache()
    free_gb, total_gb = gpu_memory_gb()
    logger.info(
        "GPU ready for inpaint (free=%.2f/%.2f GB, budget≈%.1f GB)",
        free_gb,
        total_gb,
        _inpaint_vram_budget_gb(),
    )


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
