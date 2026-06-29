"""Download and cache ML assets required by Outfit Studio."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

from outfit_studio.config import get_settings
from outfit_studio.constants import BYTES_PER_MIB
from outfit_studio.ml.checkpoints import cloth_segm_checkpoint_valid, inpaint_checkpoint_valid

logger = logging.getLogger(__name__)

CLOTH_SEGM_GDRIVE_ID = "11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"


def download_cloth_segm(models_dir: Path | None = None) -> Path:
    settings = get_settings()
    models_dir = models_dir or settings.resolved_models_dir
    dest = models_dir / settings.extra_clothes_model
    if cloth_segm_checkpoint_valid(dest):
        logger.info("U2NET cloth model present: %s", dest)
        return dest

    if dest.is_file():
        logger.warning("Removing corrupt %s", dest.name)
        dest.unlink()

    try:
        import gdown
    except ImportError as e:
        msg = "gdown is required to download cloth_segm.pth (pip install gdown)"
        raise RuntimeError(msg) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={CLOTH_SEGM_GDRIVE_ID}"
    logger.info("Downloading U2NET cloth_segm.pth from Google Drive...")
    gdown.download(url, str(dest), quiet=False)
    if not cloth_segm_checkpoint_valid(dest):
        if dest.is_file():
            dest.unlink()
        msg = f"Failed to download a valid {dest.name}"
        raise RuntimeError(msg)
    logger.info("Downloaded %s (%.1f MB)", dest, dest.stat().st_size / BYTES_PER_MIB)
    return dest


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


def warmup_segformer() -> None:
    """Pre-download HuggingFace SegFormer weights."""
    logger.info("Warming up SegFormer weights...")
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    model_id = get_settings().segformer_model
    SegformerImageProcessor.from_pretrained(model_id)
    AutoModelForSemanticSegmentation.from_pretrained(model_id)
    logger.info("SegFormer weights cached")


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

    download_cloth_segm(models_dir)
    if not skip_heavy:
        download_default_inpaint_checkpoint(models_dir)

    warmup_pose_models()
    warmup_segformer()
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
