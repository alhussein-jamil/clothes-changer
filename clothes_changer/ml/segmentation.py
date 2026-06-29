"""Shared segmentation workflow for UI and generation pipeline."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from clothes_changer.config import Settings, get_settings
from clothes_changer.ml.gpu_memory import release_inpaint_gpu, release_segmentation_gpu
from clothes_changer.ml.pipeline_debug import PipelineDebugSession
from clothes_changer.ml.segmentor import get_segmentor

logger = logging.getLogger(__name__)


def run_segmentation(
    image: Image.Image,
    *,
    settings: Settings | None = None,
    username: str = "guest",
    debug_session_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Segment *image* into person and clothes masks.

    Returns ``(person_mask, clothes_mask, active_debug_dir)``.
    """
    settings = settings or get_settings()
    image = image.convert("RGB")

    session, active_dir = PipelineDebugSession.open_or_create(settings, username, debug_session_dir)
    seg_debug = None
    if session is not None:
        seg_debug = session.subfolder("segmentation")
        seg_debug.metadata.update(
            {
                "username": username,
                "segformer_model": settings.segformer_model,
                "u2net_model": settings.extra_clothes_model,
                "image_size": list(image.size),
            }
        )
        seg_debug.save_image("00_source.png", image)

    release_inpaint_gpu()
    logger.info(
        "segment: running segmentor on %dx%d image (user=%r)",
        image.width,
        image.height,
        username,
    )
    _, person, clothes = get_segmentor().segment(image, debug=seg_debug)
    release_segmentation_gpu()
    logger.info(
        "segment: done — person_pixels=%d clothes_pixels=%d",
        int(person.sum()),
        int(clothes.sum()),
    )
    return person, clothes, active_dir
