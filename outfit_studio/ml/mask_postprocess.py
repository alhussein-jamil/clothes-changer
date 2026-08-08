"""Post-processing for semantic segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np

from outfit_studio.content_config import get_content_settings
from outfit_studio.utils.image import fill_mask_holes, grow_mask, smooth_binary_mask


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


def _refine_clothes_mask(
    person: np.ndarray,
    clothes: np.ndarray,
    *,
    min_component_area: int,
    clothes_smooth_px: int,
    clothes_edge_grow_px: int,
) -> np.ndarray:
    """Clothes-only cleanup: constrain, despeckle, fill holes, smooth, grow.

    Person silhouette is never expanded or hole-filled.
    """
    clothes = (clothes > 0).astype(np.uint8) & person
    clothes = remove_small_components(clothes, min_component_area)
    # Enclosed mid-garment gaps (pants fly, etc.), then morph-close for pinholes.
    clothes = fill_mask_holes(clothes)
    clothes = smooth_binary_mask(clothes, clothes_smooth_px) & person

    if clothes_edge_grow_px > 0 and clothes.any():
        clothes = grow_mask(clothes, clothes_edge_grow_px) & person
    return clothes


def refine_segmentation_masks(
    person_mask: np.ndarray,
    clothes_mask: np.ndarray,
    *,
    min_component_area: int | None = None,
    clothes_edge_grow_px: int | None = None,
    clothes_smooth_px: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain clothes to person; fill/smooth clothes edges. Person unchanged."""
    content = get_content_settings()
    if min_component_area is None:
        min_component_area = content.min_component_area
    if clothes_edge_grow_px is None:
        clothes_edge_grow_px = content.clothes_edge_grow_px
    if clothes_smooth_px is None:
        clothes_smooth_px = content.clothes_smooth_px

    person = (person_mask > 0).astype(np.uint8)
    clothes = _refine_clothes_mask(
        person,
        clothes_mask,
        min_component_area=min_component_area,
        clothes_smooth_px=clothes_smooth_px,
        clothes_edge_grow_px=clothes_edge_grow_px,
    )
    return person, clothes
