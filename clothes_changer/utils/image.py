"""Image and mask utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageFilter

from clothes_changer.constants import (
    BLEND_FEATHER_DIVISOR,
    BLEND_MASK_GROW_DIVISOR,
    CROP_BOX_PADDING_RATIO,
    DEFAULT_MASK_GROW_PX,
    INSTANCE_MASK_GROW_DIVISOR,
    UI,
)

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
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    px = overlay.load()
    h, w = person_mask.shape
    for y in range(h):
        for x in range(w):
            if clothes_mask[y, x]:
                px[x, y] = clothes_color
            elif person_mask[y, x]:
                px[x, y] = person_color
    return Image.alpha_composite(base, overlay)


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
    """Pad image to square using edge reflection."""
    original_width, original_height = image.size
    new_width, new_height = new_size

    aspect_ratio = original_width / original_height
    new_aspect_ratio = new_width / new_height

    if center is None:
        center = (original_width // 2, original_height // 2)

    if aspect_ratio > new_aspect_ratio:
        scaled_height = int(new_width / aspect_ratio)
        scaled_image = image.resize((new_width, scaled_height), Image.LANCZOS)
        center_y_ratio = center[1] / original_height
        adjusted_padding_top = int((new_height - scaled_height) * center_y_ratio)
        adjusted_padding_bottom = new_height - scaled_height - adjusted_padding_top

        padded_image = Image.new(image.mode, (new_width, new_height))
        padded_image.paste(scaled_image, (0, adjusted_padding_top))

        for i in range(adjusted_padding_top):
            padded_image.paste(
                scaled_image.crop((0, 0, new_width, 1)),
                (0, adjusted_padding_top - i - 1),
            )
        for i in range(adjusted_padding_bottom):
            padded_image.paste(
                scaled_image.crop((0, scaled_height - 1, new_width, scaled_height)),
                (0, new_height - adjusted_padding_bottom + i),
            )

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
        adjusted_padding_left = int((new_width - scaled_width) * center_x_ratio)
        adjusted_padding_right = new_width - scaled_width - adjusted_padding_left

        padded_image = Image.new(image.mode, (new_width, new_height))
        padded_image.paste(scaled_image, (adjusted_padding_left, 0))

        for i in range(adjusted_padding_left):
            padded_image.paste(
                scaled_image.crop((0, 0, 1, new_height)),
                (adjusted_padding_left - i - 1, 0),
            )
        for i in range(adjusted_padding_right):
            padded_image.paste(
                scaled_image.crop((scaled_width - 1, 0, scaled_width, new_height)),
                (new_width - adjusted_padding_right + i, 0),
            )

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
    max_dim = max(bottom - top, right - left)
    padding = int(CROP_BOX_PADDING_RATIO * max_dim)
    target_size = max_dim + 2 * padding
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    return {
        "left": max(0, center_x - target_size // 2),
        "top": max(0, center_y - target_size // 2),
        "right": min(mask.width, center_x + target_size // 2),
        "bottom": min(mask.height, center_y + target_size // 2),
        "center": (center_x, center_y),
    }


def prepare_instance_masks(
    person_mask: NDArray[np.uint8],
    clothes_mask: NDArray[np.uint8],
    bboxes: np.ndarray,
) -> list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
    """Split editor masks per detected person and grow mask regions.

    Assigns editor masks into detector bboxes, then grows each combined mask.
    """
    instances: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] = []
    shape = person_mask.shape
    for bbox in bboxes:
        left, top, right, bottom = bbox.astype(int)[:4]
        top, left, bottom, right = clip_bbox((top, left, bottom, right), shape)
        person = np.zeros_like(person_mask)
        clothes = np.zeros_like(clothes_mask)
        person[top:bottom, left:right] = person_mask[top:bottom, left:right]
        clothes[top:bottom, left:right] = clothes_mask[top:bottom, left:right]
        if not person.any() and not clothes.any():
            continue
        instances.append((person, clothes))

    for i, (person, clothes) in enumerate(instances):
        combined = np.logical_or(person, clothes)
        top, left, bottom, right = get_bounding_box(combined.astype(np.uint8))
        grow_amount = (bottom - top + right - left) // INSTANCE_MASK_GROW_DIVISOR
        instances[i] = (
            grow_mask(person, grow_amount),
            grow_mask(clothes, grow_amount),
        )
    return instances
