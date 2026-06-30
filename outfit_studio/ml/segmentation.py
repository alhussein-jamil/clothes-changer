"""Shared segmentation workflow for UI and generation pipeline."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from outfit_studio.config import Settings, get_settings
from outfit_studio.ml.gpu_memory import prepare_for_segmentation
from outfit_studio.ml.pipeline_debug import PipelineDebugSession
from outfit_studio.ml.segmentor import get_segmentor
from outfit_studio.operation_control import check_cancelled

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
                "human_parser_model": settings.human_parser_model,
                "image_size": list(image.size),
                "clothes_confidence": settings.segmentation_clothes_confidence,
            }
        )
        seg_debug.save_image("00_source.png", image)

    prepare_for_segmentation()
    check_cancelled()
    logger.info(
        "segment: running segmentor on %dx%d image (user=%r)",
        image.width,
        image.height,
        username,
    )
    person, clothes = get_segmentor().segment(image, debug=seg_debug)
    check_cancelled()
    logger.info(
        "segment: done — person_pixels=%d clothes_pixels=%d",
        int(person.sum()),
        int(clothes.sum()),
    )
    return person, clothes, active_dir
