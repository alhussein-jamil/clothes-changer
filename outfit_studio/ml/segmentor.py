"""Clothes segmentation — SegFormer B2 + U2NET."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from outfit_studio.config import Settings, get_settings
from outfit_studio.constants import (
    CLOTHES_SEGFORMER_CATEGORIES,
    PERSON_SEGFORMER_CATEGORIES,
    U2NET_OUTPUT_CLASSES,
)
from outfit_studio.ml.gpu_memory import (
    free_cuda_cache,
    model_load_lock,
    prefer_cpu_for_segmentation,
)
from outfit_studio.ml.pipeline_debug import PipelineDebugSession
from outfit_studio.ml.process import generate_mask, get_palette, load_seg_model
from outfit_studio.utils.logging import log_duration

logger = logging.getLogger(__name__)


class ClothesSegmentor:
    """SegFormer + U2NET fusion matching the original app."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if prefer_cpu_for_segmentation():
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("ClothesSegmentor device=%s", self.device)
        self._processor: SegformerImageProcessor | None = None
        self._model: AutoModelForSemanticSegmentation | None = None
        self._u2net: torch.nn.Module | None = None
        self._palette: list[int] | None = None
        self._lock = threading.RLock()

    def unload(self) -> None:
        """Free VRAM held by segmentation models."""
        with self._lock:
            if self._model is not None or self._u2net is not None:
                logger.info("Unloading segmentation models from %s", self.device)
            self._model = None
            self._u2net = None
            self._processor = None
            free_cuda_cache()

    def is_loaded(self) -> bool:
        return self._model is not None and self._u2net is not None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None and self._u2net is not None:
                return
            segformer_id = self.settings.segformer_model
            try:
                with model_load_lock():
                    with log_duration(logger, "load segmentation models", device=self.device):
                        logger.info(
                            "Loading SegFormer clothes model (%s) on %s...",
                            segformer_id,
                            self.device,
                        )
                        self._processor = SegformerImageProcessor.from_pretrained(segformer_id)
                        self._model = AutoModelForSemanticSegmentation.from_pretrained(
                            segformer_id,
                            low_cpu_mem_usage=False,
                        )
                        self._model = self._model.to(self.device)

                        u2net_path = (
                            self.settings.resolved_models_dir / self.settings.extra_clothes_model
                        )
                        from outfit_studio.ml.checkpoints import cloth_segm_checkpoint_valid
                        from outfit_studio.scripts.download_models import download_cloth_segm

                        if not cloth_segm_checkpoint_valid(u2net_path):
                            logger.info("U2NET weights missing or invalid — triggering download")
                            download_cloth_segm(self.settings.resolved_models_dir)
                        logger.info("Loading U2NET cloth model from %s", u2net_path)
                        self._u2net = load_seg_model(u2net_path, device=self.device)
                        self._palette = get_palette(U2NET_OUTPUT_CLASSES)
                    logger.info("Clothes segmentation models ready")
            except Exception:
                self._processor = None
                self._model = None
                self._u2net = None
                self._palette = None
                raise

    def segment_clothes(
        self,
        image: Image.Image,
        debug: PipelineDebugSession | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            return self._segment_clothes_locked(image, debug=debug)

    def _segment_clothes_locked(
        self,
        image: Image.Image,
        debug: PipelineDebugSession | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._load()
        if self._model is None or self._u2net is None or self._palette is None:
            msg = "Segmentation models failed to load"
            raise RuntimeError(msg)
        assert self._processor is not None

        w, h = image.size
        logger.debug("segment_clothes %dx%d", w, h)

        with log_duration(logger, "U2NET mask"):
            with torch.amp.autocast("cuda", enabled=self.device == "cuda"):
                extra_cloth_seg = generate_mask(
                    image,
                    net=self._u2net,
                    palette=self._palette,
                    device=self.device,
                )

        if debug is not None:
            debug.save_image("u2net/01_u2net_mask.png", extra_cloth_seg.convert("RGB"))

        extra_cloth_mask = pil_to_tensor(extra_cloth_seg)
        extra_cloth_mask = torch.where(extra_cloth_mask > 0, 1, 0)[0]

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with log_duration(logger, "SegFormer inference"):
            with torch.inference_mode():
                outputs = self._model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]

            person_mask = torch.zeros_like(pred_seg, device=self.device)
            clothes_mask = torch.zeros_like(pred_seg, device=self.device)
            for cat in PERSON_SEGFORMER_CATEGORIES:
                person_mask[pred_seg == cat] = 1
            for cat in CLOTHES_SEGFORMER_CATEGORIES:
                clothes_mask[pred_seg == cat] = 1

            if debug is not None:
                debug.save_tensor_mask("segformer/01_person_mask.png", person_mask)
                debug.save_tensor_mask("segformer/02_clothes_mask.png", clothes_mask)
                pred_np = pred_seg.cpu().numpy().astype(np.uint8)
                debug.save_image(
                    "segformer/03_label_map.png",
                    Image.fromarray(pred_np, mode="L"),
                )

            combined_clothes_mask = torch.logical_or(
                clothes_mask,
                torch.logical_and(
                    torch.logical_not(person_mask),
                    extra_cloth_mask.to(self.device),
                ),
            ).float()

            person_mask_cpu = person_mask.cpu()
            combined_clothes_mask_cpu = combined_clothes_mask.cpu()

        logger.debug(
            "segment_clothes done — person=%d clothes=%d pixels",
            int(person_mask_cpu.sum()),
            int(combined_clothes_mask_cpu.sum()),
        )

        if debug is not None:
            person_np = person_mask_cpu.numpy().astype(np.uint8)
            clothes_np = combined_clothes_mask_cpu.numpy().astype(np.uint8)
            debug.save_mask("04_fused_person_mask.png", person_np)
            debug.save_mask("05_fused_clothes_mask.png", clothes_np)
            debug.save_overlay("06_fused_overlay.png", image, person_np, clothes_np)
            debug.metadata["person_pixels"] = int(person_np.sum())
            debug.metadata["clothes_pixels"] = int(clothes_np.sum())
            debug.save_meta()

        return person_mask_cpu, combined_clothes_mask_cpu

    def segment(
        self,
        image: Image.Image,
        debug: PipelineDebugSession | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(person_mask, clothes_mask)`` as uint8 arrays."""
        person_t, clothes_t = self.segment_clothes(image.convert("RGB"), debug=debug)
        return person_t.numpy().astype(np.uint8), clothes_t.numpy().astype(np.uint8)


@lru_cache
def get_segmentor() -> ClothesSegmentor:
    return ClothesSegmentor()
