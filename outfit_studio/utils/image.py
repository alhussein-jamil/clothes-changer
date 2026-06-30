"""Image and mask utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from outfit_studio.constants import (
    BLEND_FEATHER_DIVISOR,
    BLEND_MASK_GROW_DIVISOR,
    CROP_BOX_PADDING_RATIO,
    DEFAULT_MASK_GROW_PX,
    INSTANCE_MASK_GROW_DIVISOR,
)
from outfit_studio.ui.theme import UI

if TYPE_CHECKING:
    from numpy.typing import NDArray


def resize_max(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def mask_overlay(
    image: Image.Image,
    person_mask: NDArray[np.uint8],
    clothes_mask: NDArray[np.uint8],
    person_color: tuple[int, int, int, int] = UI.PERSON_COLOR,
    clothes_color: tuple[int, int, int, int] = UI.CLOTHES_COLOR,
) -> Image.Image:
    """RGBA overlay for editor preview."""
    base = np.array(image.convert("RGBA"))
    overlay = np.zeros_like(base)
    person_on = (person_mask > 0) & ~(clothes_mask > 0)
    clothes_on = clothes_mask > 0
    overlay[person_on] = person_color
    overlay[clothes_on] = clothes_color
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    blended = (
        base[:, :, :3].astype(np.float32) * (1.0 - alpha)
        + overlay[:, :, :3].astype(np.float32) * alpha
    )
    out = np.empty_like(base)
    out[:, :, :3] = blended.astype(np.uint8)
    out[:, :, 3] = np.maximum(base[:, :, 3], overlay[:, :, 3])
    return Image.fromarray(out, mode="RGBA")


def get_bounding_box(mask: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, mask.shape[0], mask.shape[1]
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return int(top), int(left), int(bottom) + 1, int(right) + 1


def clip_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clamp (top, left, bottom, right) to valid range for array shape (h, w)."""
    top, left, bottom, right = bbox
    h, w = shape
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    bottom = max(top + 1, min(bottom, h))
    right = max(left + 1, min(right, w))
    return top, left, bottom, right


def align_masks(
    person: NDArray[np.uint8],
    clothes: NDArray[np.uint8],
    height: int,
    width: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Resize masks to match image height x width if Gradio layer size differs."""
    target = (height, width)
    if person.shape == target and clothes.shape == target:
        return person, clothes
    person = cv2.resize(person, (width, height), interpolation=cv2.INTER_NEAREST)
    clothes = cv2.resize(clothes, (width, height), interpolation=cv2.INTER_NEAREST)
    return (person > 0).astype(np.uint8), (clothes > 0).astype(np.uint8)


def grow_mask(mask: np.ndarray, amount: int = DEFAULT_MASK_GROW_PX) -> np.ndarray:
    if amount <= 0:
        return mask
    k = amount if amount % 2 == 1 else amount + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel)


def apply_reflection_padding(
    image: Image.Image,
    new_size: tuple[int, int],
    center: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict | None]:
    """Pad image to square using edge reflection.

    *center* is in crop-local coordinates (origin = top-left of *image*).
    """
    original_width, original_height = image.size
    new_width, new_height = new_size
    if original_width <= 0 or original_height <= 0 or new_width <= 0 or new_height <= 0:
        return image, None

    aspect_ratio = original_width / original_height
    new_aspect_ratio = new_width / new_height

    if center is None:
        center = (original_width // 2, original_height // 2)

    if aspect_ratio > new_aspect_ratio:
        scaled_height = int(new_width / aspect_ratio)
        scaled_image = image.resize((new_width, scaled_height), Image.LANCZOS)
        center_y_ratio = center[1] / original_height
        slack = max(0, new_height - scaled_height)
        adjusted_padding_top = int(slack * center_y_ratio)
        adjusted_padding_bottom = slack - adjusted_padding_top

        scaled_arr = np.array(scaled_image)
        pad_width = ((adjusted_padding_top, adjusted_padding_bottom), (0, 0))
        if scaled_arr.ndim == 3:
            pad_width = (*pad_width, (0, 0))
        padded_arr = np.pad(scaled_arr, pad_width, mode="edge")
        padded_image = Image.fromarray(padded_arr, mode=scaled_image.mode)

        padding_info = {
            "top": adjusted_padding_top,
            "bottom": adjusted_padding_bottom,
            "left": 0,
            "right": 0,
            "original_size": image.size,
        }
    else:
        scaled_width = int(new_height * aspect_ratio)
        scaled_image = image.resize((scaled_width, new_height), Image.LANCZOS)
        center_x_ratio = center[0] / original_width
        slack = max(0, new_width - scaled_width)
        adjusted_padding_left = int(slack * center_x_ratio)
        adjusted_padding_right = slack - adjusted_padding_left

        scaled_arr = np.array(scaled_image)
        pad_width = ((0, 0), (adjusted_padding_left, adjusted_padding_right))
        if scaled_arr.ndim == 3:
            pad_width = (*pad_width, (0, 0))
        padded_arr = np.pad(scaled_arr, pad_width, mode="edge")
        padded_image = Image.fromarray(padded_arr, mode=scaled_image.mode)

        padding_info = {
            "top": 0,
            "bottom": 0,
            "left": adjusted_padding_left,
            "right": adjusted_padding_right,
            "original_size": image.size,
        }

    return padded_image, padding_info


def remove_reflection_padding(padded_image: Image.Image, padding_info: dict | None) -> Image.Image:
    if padding_info is None:
        return padded_image
    padded_width, padded_height = padded_image.size
    original_width, original_height = padding_info["original_size"]
    unpadded_image = padded_image.crop(
        (
            padding_info["left"],
            padding_info["top"],
            padded_width - padding_info["right"],
            padded_height - padding_info["bottom"],
        )
    )
    return unpadded_image.resize((original_width, original_height), Image.LANCZOS)


def grow_mask_pil(mask: Image.Image, grow_amount: int) -> Image.Image:
    if grow_amount % 2 == 0:
        grow_amount += 1
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_amount, grow_amount))
    grown = cv2.dilate(mask_np.astype(np.uint8), kernel)
    return Image.fromarray(grown)


def feather_mask_pil(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius))


def blend_images_with_enhancements(
    original: Image.Image,
    inpainted: Image.Image,
    clothes_mask: Image.Image,
    person_mask: Image.Image,
) -> Image.Image:
    """Feathered alpha blend for inpainted regions."""
    clothes_np = np.array(clothes_mask) > 0
    if not clothes_np.any():
        return original.convert("RGBA")

    top, left, bottom, right = get_bounding_box(clothes_np.astype(np.uint8))
    grow_amount = max(bottom - top, right - left) // BLEND_MASK_GROW_DIVISOR

    grown = grow_mask_pil(clothes_mask, grow_amount)
    feathered_mask = feather_mask_pil(grown, max(1, grow_amount // BLEND_FEATHER_DIVISOR))

    # Suppress feather only in true background (outside person + clothes).
    person_np = np.array(person_mask.convert("L")) > 0
    clothes_np_bool = clothes_np
    body = person_np | clothes_np_bool
    if body.any():
        feather_np = np.array(feathered_mask, dtype=np.float32)
        feather_np = np.where(body, feather_np, 0)
        feathered_mask = Image.fromarray(feather_np.astype(np.uint8), mode="L")

    inpainted_rgba = inpainted.convert("RGBA")
    inpainted_rgba.putalpha(feathered_mask)
    return Image.alpha_composite(original.convert("RGBA"), inpainted_rgba)


def composite_crop_onto(
    full_image: Image.Image,
    patch: Image.Image,
    left: int,
    top: int,
) -> Image.Image:
    """Alpha-composite an inpainted crop onto the full frame (no hard paste seam)."""
    base = full_image.convert("RGBA")
    patch_rgba = patch.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(patch_rgba, (left, top))
    return Image.alpha_composite(base, layer).convert("RGB")


def get_crop_info(mask: Image.Image) -> dict:
    top, left, bottom, right = get_bounding_box(np.array(mask) > 0)
    max_dim = max(bottom - top, right - left, 1)
    padding = int(CROP_BOX_PADDING_RATIO * max_dim)
    target_size = max(max_dim + 2 * padding, 1)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    half_lo = target_size // 2
    half_hi = target_size - half_lo
    crop_left = max(0, center_x - half_lo)
    crop_top = max(0, center_y - half_lo)
    crop_right = min(mask.width, center_x + half_hi)
    crop_bottom = min(mask.height, center_y + half_hi)
    if crop_right <= crop_left:
        crop_right = min(mask.width, crop_left + 1)
    if crop_bottom <= crop_top:
        crop_bottom = min(mask.height, crop_top + 1)
    return {
        "left": crop_left,
        "top": crop_top,
        "right": crop_right,
        "bottom": crop_bottom,
        "center": (center_x - crop_left, center_y - crop_top),
    }


def _bbox_marker_points(
    combined: NDArray[np.uint8],
    bboxes: np.ndarray,
    distance: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """Return watershed seeds at the thickest interior point of each xyxy bbox."""
    h, w = combined.shape
    foreground = combined > 0
    foreground_coords = np.column_stack(np.where(foreground))
    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        region = distance[y1:y2, x1:x2].copy()
        region[~foreground[y1:y2, x1:x2]] = -1
        if region.max() > 0:
            peak = np.unravel_index(int(np.argmax(region)), region.shape)
            row, col = y1 + int(peak[0]), x1 + int(peak[1])
        else:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            if foreground[cy, cx]:
                row, col = cy, cx
            elif foreground_coords.size == 0:
                continue
            else:
                dist = (foreground_coords[:, 0] - cy) ** 2 + (foreground_coords[:, 1] - cx) ** 2
                nearest = foreground_coords[int(np.argmin(dist))]
                row, col = int(nearest[0]), int(nearest[1])

        if (row, col) in seen:
            continue
        seen.add((row, col))
        points.append((row, col))
    return points


def _watershed_instance_labels(
    combined: NDArray[np.uint8],
    bboxes: np.ndarray,
) -> NDArray[np.int32] | None:
    """Partition a merged semantic mask into one label per detected person."""
    foreground = combined > 0
    if not foreground.any():
        return None

    distance = ndi.distance_transform_edt(foreground)
    marker_points = _bbox_marker_points(combined, bboxes, distance)
    if not marker_points:
        return None

    markers = np.zeros(combined.shape, dtype=np.int32)
    for label_id, (row, col) in enumerate(marker_points, start=1):
        markers[row, col] = label_id

    return watershed(-distance, markers, mask=foreground)


def _grow_instance_masks(
    instances: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
) -> list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
    grown: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] = []
    for person, clothes in instances:
        combined = np.logical_or(person, clothes)
        top, left, bottom, right = get_bounding_box(combined.astype(np.uint8))
        grow_amount = (bottom - top + right - left) // INSTANCE_MASK_GROW_DIVISOR
        grown.append((grow_mask(person, grow_amount), grow_mask(clothes, grow_amount)))
    return grown


def prepare_instance_masks(
    person_mask: NDArray[np.uint8],
    clothes_mask: NDArray[np.uint8],
    bboxes: np.ndarray,
) -> list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
    """Split semantic masks into per-person instances and grow mask regions.

    The human parser outputs one merged person/clothes mask for the whole image. When
    multiple people are present, YOLOX bboxes seed a marker-controlled
    watershed on the combined mask. Markers are placed at the thickest interior
    point inside each bbox (distance-transform peak), not at the bbox center.
    """
    if len(bboxes) == 0:
        return []

    combined = np.logical_or(person_mask > 0, clothes_mask > 0)
    if not combined.any():
        return []

    if len(bboxes) == 1:
        x1, y1, x2, y2 = bboxes[0][:4].astype(int)
        top, left, bottom, right = clip_bbox((y1, x1, y2, x2), person_mask.shape)
        person = np.zeros_like(person_mask)
        clothes = np.zeros_like(clothes_mask)
        person[top:bottom, left:right] = person_mask[top:bottom, left:right]
        clothes[top:bottom, left:right] = clothes_mask[top:bottom, left:right]
        if not person.any() and not clothes.any():
            return []
        return _grow_instance_masks([(person, clothes)])

    labels = _watershed_instance_labels(combined.astype(np.uint8), bboxes)
    if labels is None:
        return []

    instances: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] = []
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not region.any():
            continue
        person = (person_mask > 0) & region
        clothes = (clothes_mask > 0) & region
        if not person.any() and not clothes.any():
            continue
        instances.append((person.astype(np.uint8), clothes.astype(np.uint8)))

    return _grow_instance_masks(instances)
