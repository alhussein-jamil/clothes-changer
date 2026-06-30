"""Download and cache ML assets required by Outfit Studio."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

from outfit_studio.config import get_settings
from outfit_studio.ml.checkpoints import inpaint_checkpoint_valid

logger = logging.getLogger(__name__)


def download_default_inpaint_checkpoint(models_dir: Path | None = None) -> Path | None:
    """Download or cache the configured default inpaint model."""
    from outfit_studio.ml.checkpoints import is_hub_model_id
    from outfit_studio.ml.inpainter import InpaintEngine

    settings = get_settings()
    models_dir = models_dir or settings.resolved_models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    engine = InpaintEngine()
    model_id = engine.default_model_id()
    if is_hub_model_id(model_id):
        logger.info("Caching Hugging Face inpaint model: %s", model_id)
        engine.load(model_id)
        engine.warmup()
        return None
    path = models_dir / model_id
    if inpaint_checkpoint_valid(path):
        logger.info("Default inpaint model present: %s", path)
        return path
    logger.info("Downloading default inpaint model: %s", model_id)
    engine.download_model(path)
    return path


def warmup_pose_models() -> None:
    """Pre-download rtmlib ONNX weights via a dummy inference."""
    import torch

    from outfit_studio.ml.pose import get_pose_estimator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Warming up pose/detector ONNX models (device=%s)...", device)
    est = get_pose_estimator()
    dummy = Image.new("RGB", (256, 256), color=(128, 128, 128))
    est.get_bboxes(dummy)
    est.estimate(dummy)
    logger.info("Pose models cached")


def warmup_human_parser() -> None:
    """Pre-download Hugging Face human parser weights."""
    logger.info("Warming up human parser weights...")
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    model_id = get_settings().human_parser_model
    SegformerImageProcessor.from_pretrained(model_id)
    AutoModelForSemanticSegmentation.from_pretrained(model_id)
    logger.info("Human parser weights cached")


def warmup_controlnet() -> None:
    """Pre-download ControlNet OpenPose weights."""
    import torch
    from diffusers import ControlNetModel

    settings = get_settings()
    logger.info("Warming up ControlNet: %s", settings.controlnet_model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ControlNetModel.from_pretrained(settings.controlnet_model, torch_dtype=dtype)
    logger.info("ControlNet weights cached")


def download_all(*, skip_heavy: bool = False) -> None:
    settings = get_settings()
    settings.ensure_dirs()
    models_dir = settings.resolved_models_dir
    logger.info("Starting model download (skip_heavy=%s)", skip_heavy)

    if not skip_heavy:
        download_default_inpaint_checkpoint(models_dir)

    warmup_pose_models()
    warmup_human_parser()
    if settings.use_controlnet:
        warmup_controlnet()
    logger.info("All model assets ready")


def main() -> None:
    from outfit_studio.ml.onnx_runtime import ensure_nvidia_cuda_libs
    from outfit_studio.utils import setup_logging

    setup_logging()
    ensure_nvidia_cuda_libs()
    skip_heavy = "--skip-heavy" in sys.argv
    download_all(skip_heavy=skip_heavy)


if __name__ == "__main__":
    main()
