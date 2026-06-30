"""Clothes segmentation via FASHN Human Parser."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from outfit_studio.config import Settings, get_settings
from outfit_studio.ml.gpu_memory import (
    free_cuda_cache,
    model_load_lock,
    prefer_cpu_for_segmentation,
)
from outfit_studio.ml.mask_postprocess import refine_segmentation_masks
from outfit_studio.ml.parser_labels import masks_from_parser_logits
from outfit_studio.ml.pipeline_debug import PipelineDebugSession
from outfit_studio.utils.logging import log_duration

logger = logging.getLogger(__name__)


class ClothesSegmentor:
    """Fashion-tuned human parser with confidence-aware clothing masks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if prefer_cpu_for_segmentation():
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("ClothesSegmentor device=%s", self.device)
        self._processor: SegformerImageProcessor | None = None
        self._model: AutoModelForSemanticSegmentation | None = None
        self._lock = threading.RLock()

    def unload(self) -> None:
        """Free VRAM held by the human parser."""
        with self._lock:
            if self._model is not None:
                logger.info("Unloading human parser from %s", self.device)
            self._model = None
            self._processor = None
            free_cuda_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            model_id = self.settings.content.human_parser
            try:
                with model_load_lock():
                    with log_duration(logger, "load human parser", device=self.device):
                        logger.info("Loading human parser (%s) on %s...", model_id, self.device)
                        self._processor = SegformerImageProcessor.from_pretrained(model_id)
                        self._model = AutoModelForSemanticSegmentation.from_pretrained(
                            model_id,
                            low_cpu_mem_usage=False,
                        )
                        self._model = self._model.to(self.device)
                    logger.info("Human parser ready")
            except Exception:
                self._processor = None
                self._model = None
                raise

    def segment(
        self,
        image: Image.Image,
        debug: PipelineDebugSession | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(person_mask, clothes_mask)`` as uint8 arrays."""
        with self._lock:
            return self._segment_locked(image.convert("RGB"), debug=debug)

    def _segment_locked(
        self,
        image: Image.Image,
        debug: PipelineDebugSession | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._load()
        if self._model is None:
            msg = "Human parser failed to load"
            raise RuntimeError(msg)
        assert self._processor is not None

        w, h = image.size
        logger.debug("segment %dx%d", w, h)

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with log_duration(logger, "human parser inference"):
            with torch.inference_mode():
                outputs = self._model(**inputs)
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            person_mask, clothes_mask = masks_from_parser_logits(
                upsampled_logits,
                confidence=self.settings.content.clothes_confidence,
            )

            if debug is not None:
                pred_seg = upsampled_logits.argmax(dim=1)[0]
                debug.save_tensor_mask("parser/01_person_mask.png", person_mask)
                debug.save_tensor_mask("parser/02_clothes_mask.png", clothes_mask)
                pred_np = pred_seg.cpu().numpy().astype(np.uint8)
                debug.save_image(
                    "parser/03_label_map.png",
                    Image.fromarray(pred_np, mode="L"),
                )

            person_np = (person_mask > 0).to(torch.uint8).cpu().numpy()
            clothes_np = (clothes_mask > 0).to(torch.uint8).cpu().numpy()
            person_np, clothes_np = refine_segmentation_masks(
                person_np,
                clothes_np,
                min_component_area=self.settings.content.min_component_area,
                clothes_edge_grow_px=self.settings.content.clothes_edge_grow_px,
            )

        logger.debug(
            "segment done — person=%d clothes=%d pixels",
            int(person_np.sum()),
            int(clothes_np.sum()),
        )

        if debug is not None:
            debug.save_mask("04_refined_person_mask.png", person_np)
            debug.save_mask("05_refined_clothes_mask.png", clothes_np)
            debug.save_overlay("06_refined_overlay.png", image, person_np, clothes_np)
            debug.metadata["person_pixels"] = int(person_np.sum())
            debug.metadata["clothes_pixels"] = int(clothes_np.sum())
            debug.save_meta()

        return person_np, clothes_np


@lru_cache
def get_segmentor() -> ClothesSegmentor:
    return ClothesSegmentor()
