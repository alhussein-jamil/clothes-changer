"""Post-processing for semantic segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np

from outfit_studio.constants import (
    SEGMENTATION_CLOTHES_EDGE_GROW_PX,
    SEGMENTATION_MIN_COMPONENT_AREA,
)
from outfit_studio.utils.image import grow_mask


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than *min_area* pixels."""
    if min_area <= 0 or not mask.any():
        return mask.astype(np.uint8)

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    keep = stats[1:, cv2.CC_STAT_AREA] >= min_area
    kept_labels = np.nonzero(keep)[0] + 1
    return np.isin(labels, kept_labels).astype(np.uint8)


def refine_segmentation_masks(
    person_mask: np.ndarray,
    clothes_mask: np.ndarray,
    *,
    min_component_area: int = SEGMENTATION_MIN_COMPONENT_AREA,
    clothes_edge_grow_px: int = SEGMENTATION_CLOTHES_EDGE_GROW_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain clothes to the person silhouette, drop speckle, grow edges."""
    person = (person_mask > 0).astype(np.uint8)
    clothes = (clothes_mask > 0).astype(np.uint8)

    clothes = clothes & person
    clothes = remove_small_components(clothes, min_component_area)

    if clothes_edge_grow_px > 0 and clothes.any():
        clothes = grow_mask(clothes, clothes_edge_grow_px) & person

    return person, clothes
