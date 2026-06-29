"""Pose + person detection via rtmlib ONNX (no mmcv/mmpose/mmdet)."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from rtmlib import YOLOX, RTMPose, Wholebody, draw_skeleton

from outfit_studio.config import Settings, get_settings
from outfit_studio.ml.onnx_runtime import resolve_onnx_device

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Top-down whole-body pose with OpenPose skeleton for ControlNet."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._det: YOLOX | None = None
        self._pose: RTMPose | None = None
        self.device = resolve_onnx_device()
        self.backend = "onnxruntime"
        logger.info(
            "PoseEstimator configured (device=%s, mode=%s)",
            self.device,
            self.settings.pose_mode,
        )

    def _load(self) -> None:
        if self._det is not None:
            return
        mode = self.settings.pose_mode
        cfg = Wholebody.MODE[mode]
        logger.info("Loading pose models (rtmlib %s, device=%s)...", mode, self.device)
        self._det = YOLOX(
            cfg["det"],
            model_input_size=cfg["det_input_size"],
            mode="human",
            score_thr=self.settings.detection_threshold,
            nms_thr=self.settings.detection_threshold,
            backend=self.backend,
            device=self.device,
        )
        self._pose = RTMPose(
            cfg["pose"],
            model_input_size=cfg["pose_input_size"],
            to_openpose=True,
            backend=self.backend,
            device=self.device,
        )
        logger.info("Pose models ready")

    def unload(self) -> None:
        """Release ONNX sessions so inpainting can use VRAM."""
        if self._det is not None or self._pose is not None:
            logger.info("Unloading pose/detector ONNX models")
        self._det = None
        self._pose = None
        from outfit_studio.ml.gpu_memory import free_cuda_cache

        free_cuda_cache()

    def get_bboxes(self, image: Image.Image) -> np.ndarray:
        """Person bounding boxes in xyxy format (matches original RTMDet flow)."""
        self._load()
        assert self._det is not None
        img = np.array(image.convert("RGB"))
        if img.shape[0] < 1 or img.shape[1] < 1:
            logger.warning(
                "Image too small for detection (%dx%d) — using full image bbox",
                image.width,
                image.height,
            )
            return np.array([[0, 0, max(image.width, 1), max(image.height, 1)]], dtype=np.float32)
        logger.debug("Detecting persons in %dx%d image", image.width, image.height)
        try:
            with torch.inference_mode():
                bboxes = self._det(img)
        except ZeroDivisionError:
            logger.exception("Detection failed; using full image bbox")
            return np.array([[0, 0, image.width, image.height]], dtype=np.float32)

        if bboxes is None or len(bboxes) == 0:
            logger.warning("No persons detected — using full image bbox")
            return np.array([[0, 0, image.width, image.height]], dtype=np.float32)
        logger.debug("Detected %d person(s)", len(bboxes))
        return np.asarray(bboxes, dtype=np.float32)

    def estimate(
        self,
        image: Image.Image,
        bboxes: np.ndarray | None = None,
    ) -> Image.Image:
        """OpenPose skeleton on black background for ControlNet conditioning."""
        self._load()
        assert self._pose is not None
        img = np.array(image.convert("RGB"))
        if bboxes is None:
            bboxes = self.get_bboxes(image)

        logger.debug("Estimating pose for %d bbox(es)", len(bboxes))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            keypoints, scores = self._pose(img, bboxes=bboxes)

        canvas = np.zeros_like(img)
        pose_arr = draw_skeleton(
            canvas,
            keypoints,
            scores,
            openpose_skeleton=True,
            kpt_thr=self.settings.pose_keypoint_threshold,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Pose skeleton rendered %dx%d", pose_arr.shape[1], pose_arr.shape[0])
        return Image.fromarray(pose_arr)


@lru_cache
def get_pose_estimator() -> PoseEstimator:
    return PoseEstimator()
