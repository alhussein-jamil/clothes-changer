"""Hand region masks from OpenPose-format whole-body keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from outfit_studio.constants import MASK_OFF, MASK_ON

# RTMPose whole-body OpenPose layout (134 keypoints).
OPENPOSE_LEFT_HAND: tuple[int, ...] = tuple(range(92, 113))
OPENPOSE_RIGHT_HAND: tuple[int, ...] = tuple(range(113, 134))
HAND_INDEX_GROUPS: tuple[tuple[int, ...], ...] = (OPENPOSE_LEFT_HAND, OPENPOSE_RIGHT_HAND)
MIN_HAND_KEYPOINTS: int = 8


@dataclass(frozen=True)
class HandRegion:
    """One hand extracted from a whole-body pose estimate."""

    keypoints: np.ndarray  # (21, 2) float
    scores: np.ndarray  # (21,) float
    bbox: tuple[int, int, int, int]  # left, top, right, bottom (exclusive)


def _visible_points(
    keypoints: np.ndarray,
    scores: np.ndarray,
    indices: tuple[int, ...],
    kpt_thr: float,
) -> np.ndarray:
    pts: list[np.ndarray] = []
    for idx in indices:
        if idx >= len(scores) or scores[idx] < kpt_thr:
            continue
        x, y = keypoints[idx]
        if x <= 0 and y <= 0:
            continue
        pts.append(np.array([x, y], dtype=np.float32))
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack(pts, axis=0)


def hand_regions_from_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    kpt_thr: float,
    padding_ratio: float = 0.35,
    image_size: tuple[int, int] | None = None,
) -> list[HandRegion]:
    """Return left/right hand regions when enough keypoints are visible."""
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]

    regions: list[HandRegion] = []
    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None

    for indices in HAND_INDEX_GROUPS:
        hand_kp = np.zeros((len(indices), 2), dtype=np.float32)
        hand_sc = np.zeros(len(indices), dtype=np.float32)
        visible = 0
        for out_i, src_i in enumerate(indices):
            if src_i >= len(scores):
                continue
            hand_kp[out_i] = keypoints[src_i]
            hand_sc[out_i] = scores[src_i]
            if scores[src_i] >= kpt_thr:
                visible += 1
        if visible < MIN_HAND_KEYPOINTS:
            continue

        pts = _visible_points(keypoints, scores, indices, kpt_thr)
        xs = pts[:, 0]
        ys = pts[:, 1]
        span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 24.0)
        pad = max(8, int(span * padding_ratio))
        left = int(max(0, np.floor(xs.min()) - pad))
        top = int(max(0, np.floor(ys.min()) - pad))
        right = int(np.ceil(xs.max()) + pad)
        bottom = int(np.ceil(ys.max()) + pad)
        if width is not None:
            right = min(width, right)
        if height is not None:
            bottom = min(height, bottom)
        if right <= left or bottom <= top:
            continue
        regions.append(
            HandRegion(
                keypoints=hand_kp,
                scores=hand_sc,
                bbox=(left, top, right, bottom),
            )
        )
    return regions


def mask_from_hand_region(
    region: HandRegion,
    shape_hw: tuple[int, int],
    *,
    kpt_thr: float,
    padding_ratio: float = 0.35,
) -> np.ndarray:
    """Binary uint8 mask covering one hand (255 inside, 0 outside)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = _visible_points(region.keypoints, region.scores, tuple(range(21)), kpt_thr)
    if len(pts) < 3:
        left, top, right, bottom = region.bbox
        mask[top:bottom, left:right] = MASK_ON
        return mask

    hull = cv2.convexHull(pts.astype(np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), MASK_ON)
    span = max(
        float(pts[:, 0].max() - pts[:, 0].min()),
        float(pts[:, 1].max() - pts[:, 1].min()),
        24.0,
    )
    radius = max(3, int(span * padding_ratio / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def build_combined_hand_mask(
    keypoints: np.ndarray,
    scores: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    kpt_thr: float,
    padding_ratio: float = 0.35,
) -> np.ndarray:
    """Union mask for all visible hands in an image."""
    h, w = shape_hw
    combined = np.zeros((h, w), dtype=np.uint8)
    regions = hand_regions_from_pose(
        keypoints,
        scores,
        kpt_thr=kpt_thr,
        padding_ratio=padding_ratio,
        image_size=(w, h),
    )
    for region in regions:
        combined = np.maximum(
            combined,
            mask_from_hand_region(
                region,
                shape_hw,
                kpt_thr=kpt_thr,
                padding_ratio=padding_ratio,
            ),
        )
    return combined


def subtract_hand_mask(
    clothes_mask: np.ndarray,
    hand_mask: np.ndarray,
) -> np.ndarray:
    """Remove hand pixels from a clothes mask (values 0/1 or 0/255)."""
    if clothes_mask.dtype != np.uint8:
        base = (clothes_mask > 0).astype(np.uint8)
    else:
        base = (clothes_mask > 0).astype(np.uint8)
    protected = hand_mask > 0
    result = base.copy()
    result[protected] = MASK_OFF
    return result
