"""ImageEditor mask encoding for Gradio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from gradio_client import handle_file
from PIL import Image

from outfit_studio.ui.theme import (
    CLOTHES_COLOR,
    EDITOR_CANVAS_SIZE,
    PERSON_COLOR,
    UI,
    MaskEditor,
)
from outfit_studio.utils.image import align_masks, mask_overlay

logger = logging.getLogger(__name__)


def _describe_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, Image.Image):
        return f"PIL.Image({value.mode}, {value.size})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, str):
        preview_len = UI.LOG_PREVIEW_LEN
        suffix = "..." if len(value) > preview_len else ""
        return f"str({value[:preview_len]!r}{suffix})"
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = {k: _describe_value(value[k]) for k in keys[: UI.DESCRIBE_DICT_KEYS_PREVIEW]}
        return f"dict(keys={keys}, preview={preview})"
    if isinstance(value, list | tuple):
        return f"{type(value).__name__}(len={len(value)})"
    return f"{type(value).__name__}({value!r})"


def _resolve_file_path(raw: str | dict) -> str | None:
    if isinstance(raw, dict):
        raw = raw.get("path") or raw.get("url")
    if not raw:
        return None
    resolved = handle_file(str(raw))
    if isinstance(resolved, dict):
        resolved = resolved.get("path")
    return str(resolved) if resolved else None


def _load_background_image(bg_raw: Any) -> Image.Image | None:
    """Normalize ImageEditor background payloads from Gradio."""
    if bg_raw is None:
        return None
    if isinstance(bg_raw, Image.Image):
        return bg_raw.convert("RGBA")
    if isinstance(bg_raw, np.ndarray):
        return Image.fromarray(bg_raw).convert("RGBA")
    path = _resolve_file_path(bg_raw) if isinstance(bg_raw, str | dict) else None
    if path:
        return Image.open(path).convert("RGBA")
    logger.warning(
        "parse_editor_masks: unsupported background type %s",
        _describe_value(bg_raw),
    )
    return None


def _load_layer_image(layer_raw: Any, width: int, height: int) -> Image.Image | None:
    """Load mask layer from PIL, ndarray, path, or Gradio FileData."""
    if layer_raw is None:
        return None
    if isinstance(layer_raw, np.ndarray):
        img = Image.fromarray(layer_raw, mode="RGBA")
    elif isinstance(layer_raw, Image.Image):
        img = layer_raw
    else:
        path = _resolve_file_path(layer_raw) if isinstance(layer_raw, str | dict) else None
        if not path:
            logger.warning(
                "parse_editor_masks: unsupported layer type %s",
                _describe_value(layer_raw),
            )
            return None
        img = Image.open(path)

    img = img.convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.NEAREST)
    return img


def _masks_from_layer(layer_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Read person/clothes masks from an ImageEditor layer (handles Gradio round-trips)."""
    arr = np.array(layer_img.convert("RGBA"))
    alpha = arr[..., 3]
    visible = alpha > MaskEditor.ALPHA_VISIBLE_MIN
    red = arr[..., 0].astype(np.int16)
    green = arr[..., 1].astype(np.int16)
    person = (visible & (red > MaskEditor.CHANNEL_MIN) & (red >= green)).astype(np.uint8)
    clothes = (visible & (green > MaskEditor.CHANNEL_MIN) & (green > red)).astype(np.uint8)
    return person, clothes


def _masks_from_composite(
    bg: Image.Image, composite_raw: Any, width: int, height: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Recover masks when Gradio sends composite preview but empty layers."""
    comp = _load_background_image(composite_raw)
    if comp is None:
        return None, None

    comp_rgb = comp.convert("RGB")
    if comp_rgb.size != (width, height):
        comp_rgb = comp_rgb.resize((width, height), Image.Resampling.NEAREST)

    bg_arr = np.array(bg, dtype=np.int16)
    comp_arr = np.array(comp_rgb, dtype=np.int16)
    diff = comp_arr - bg_arr
    changed = np.abs(diff).max(axis=2) > MaskEditor.COMPOSITE_DIFF_MIN

    bias = MaskEditor.COMPOSITE_CHANNEL_BIAS_MIN
    clothes = (((diff[:, :, 1] - diff[:, :, 0]) > bias) & changed).astype(np.uint8)
    person = (((diff[:, :, 0] - diff[:, :, 1]) > bias) & changed).astype(np.uint8)

    overlap = (person > 0) & (clothes > 0)
    person[overlap & (diff[:, :, 1] >= diff[:, :, 0])] = 0
    clothes[overlap & (diff[:, :, 0] > diff[:, :, 1])] = 0

    if person.sum() == 0 and clothes.sum() == 0:
        return None, None
    return person, clothes


def file_path_from_editor(editor: dict | None) -> str | None:
    """Extract a local file path from an ImageEditor value dict."""
    if not editor or not isinstance(editor, dict):
        return None
    for field in ("background", "composite"):
        raw = editor.get(field)
        if raw is None:
            continue
        path = _resolve_file_path(raw) if isinstance(raw, str | dict) else None
        if path:
            return path
    return None


def image_from_segment_key(key: str | None) -> Image.Image | None:
    """Reload a gallery/history image when Gradio drops the editor background."""
    if not key or not key.startswith("path:"):
        return None
    path = Path(key.removeprefix("path:"))
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def background_key_from_path(path: str | Path) -> str:
    """Stable segment key for images loaded from a file path (gallery, examples)."""
    return f"path:{Path(path).resolve()}"


def background_key_from_image(image: Image.Image) -> str:
    """Content fingerprint for deduplicating auto-segment on ImageEditor upload loops."""
    small = image.convert("RGB").resize(MaskEditor.FINGERPRINT_SIZE, Image.Resampling.BILINEAR)
    return f"rgb:{hash(small.tobytes())}"


def load_editor_clean_image(editor: dict | None) -> Image.Image | None:
    """Load the unmasked photo from the editor (never the composite mask preview)."""
    if not editor or not isinstance(editor, dict):
        return None
    bg_raw = editor.get("background")
    if bg_raw is None:
        comp = editor.get("composite")
        # File-backed composite only — PIL/RGBA composite includes baked-in mask colors.
        if isinstance(comp, str | dict):
            bg_raw = comp
        else:
            return None
    bg = _load_background_image(bg_raw)
    return bg.convert("RGB") if bg is not None else None


def masks_have_pixels(
    person: np.ndarray | None,
    clothes: np.ndarray | None,
) -> bool:
    return (
        person is not None
        and clothes is not None
        and (int(person.sum()) > 0 or int(clothes.sum()) > 0)
    )


def letterbox_to_editor_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int] = EDITOR_CANVAS_SIZE,
) -> Image.Image:
    """Letterbox *image* into the ImageEditor canvas (matches Gradio upload layout)."""
    cw, ch = canvas_size
    img = image.convert("RGBA")
    w, h = img.size
    if (w, h) == (cw, ch):
        return img
    scale = min(cw / w, ch / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    ox = (cw - new_w) // 2
    oy = (ch - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def letterbox_masks(
    person: np.ndarray,
    clothes: np.ndarray,
    src_size: tuple[int, int],
    canvas_size: tuple[int, int] = EDITOR_CANVAS_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same letterbox transform as ``letterbox_to_editor_canvas`` to masks."""
    sw, sh = src_size
    cw, ch = canvas_size
    if (sw, sh) == (cw, ch):
        return person, clothes

    scale = min(cw / sw, ch / sh)
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    ox = (cw - new_w) // 2
    oy = (ch - new_h) // 2

    person_r = cv2.resize(person, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    clothes_r = cv2.resize(clothes, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    person_out = np.zeros((ch, cw), dtype=np.uint8)
    clothes_out = np.zeros((ch, cw), dtype=np.uint8)
    person_out[oy : oy + new_h, ox : ox + new_w] = person_r
    clothes_out[oy : oy + new_h, ox : ox + new_w] = clothes_r
    return person_out, clothes_out


def unletterbox_masks(
    person: np.ndarray,
    clothes: np.ndarray,
    src_size: tuple[int, int],
    canvas_size: tuple[int, int] = EDITOR_CANVAS_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Map canvas-fitted masks back to the original image dimensions."""
    sw, sh = src_size
    cw, ch = canvas_size
    if person.shape == (sh, sw):
        return person, clothes

    scale = min(cw / sw, ch / sh)
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    ox = (cw - new_w) // 2
    oy = (ch - new_h) // 2

    person_crop = person[oy : oy + new_h, ox : ox + new_w]
    clothes_crop = clothes[oy : oy + new_h, ox : ox + new_w]
    person_out = cv2.resize(person_crop, (sw, sh), interpolation=cv2.INTER_NEAREST)
    clothes_out = cv2.resize(clothes_crop, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return (person_out > 0).astype(np.uint8), (clothes_out > 0).astype(np.uint8)


def editor_mask_reset(editor: dict | None, clean: Image.Image) -> dict:
    """Return a mask-free ImageEditor value at the current canvas size.

    Gradio's ImageEditor can append programmatic mask layers instead of
    replacing them; callers should push this payload before re-segmenting.
    """
    clean_rgba = clean.convert("RGBA")
    canvas = clean_rgba
    if editor and isinstance(editor, dict) and editor.get("background") is not None:
        existing = _load_background_image(editor["background"])
        if existing is not None and existing.size != clean_rgba.size:
            canvas = letterbox_to_editor_canvas(clean, existing.size)
    return {
        "background": canvas,
        "layers": [],
        "composite": canvas.convert("RGB"),
    }


def apply_masks_to_editor(
    background: Image.Image,
    person: np.ndarray,
    clothes: np.ndarray,
    editor: dict | None = None,
    *,
    clean: Image.Image | None = None,
) -> dict:
    """Build ImageEditor value with visible mask overlay for Gradio.

    Keep the editor's existing background handle when present so the client
    canvas and mask layer stay aligned. Set ``composite`` to the colored
    preview — Gradio 5 does not reliably paint programmatic ``layers`` alone.
    """
    display_bg: Image.Image | None = None
    if editor and isinstance(editor, dict) and editor.get("background") is not None:
        display_bg = _load_background_image(editor["background"])

    if display_bg is None:
        display_bg = background.convert("RGBA")

    cw, ch = display_bg.size
    person, clothes = align_masks(person, clothes, ch, cw)

    layer_arr = np.zeros((ch, cw, 4), dtype=np.uint8)
    person_only = (person > 0) & ~(clothes > 0)
    layer_arr[person_only] = PERSON_COLOR
    layer_arr[clothes > 0] = CLOTHES_COLOR
    layer_pil = Image.fromarray(layer_arr, mode="RGBA")

    overlay_base = (
        clean.convert("RGB")
        if clean is not None
        else load_editor_clean_image(editor) or display_bg.convert("RGB")
    )
    if overlay_base.size != (cw, ch):
        if clean is not None:
            overlay_base = letterbox_to_editor_canvas(clean, (cw, ch)).convert("RGB")
        else:
            overlay_base = overlay_base.resize((cw, ch), Image.Resampling.LANCZOS)

    composite = mask_overlay(overlay_base, person, clothes)

    return {
        "background": display_bg,
        "layers": [layer_pil],
        "composite": composite,
    }


def _fit_masks_to_source(
    person: np.ndarray,
    clothes: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map masks to pipeline source dimensions (letterbox-aware)."""
    if person.shape == (height, width):
        return person, clothes
    if person.shape == EDITOR_CANVAS_SIZE[::-1]:
        return unletterbox_masks(person, clothes, (width, height))
    return align_masks(person, clothes, height, width)


def resolve_masks_for_generate(
    editor: dict | None,
    segment_masks: tuple[np.ndarray, np.ndarray] | None,
    pipeline_source: Image.Image,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve person/clothes masks for inpainting at pipeline resolution."""
    width, height = pipeline_source.size
    _, editor_person, editor_clothes = parse_editor_masks(editor)
    editor_has = masks_have_pixels(editor_person, editor_clothes)
    layers = (editor or {}).get("layers") or []

    if editor_has and layers:
        return _fit_masks_to_source(editor_person, editor_clothes, width, height)

    if segment_masks is not None:
        cached_person, cached_clothes = segment_masks
        if masks_have_pixels(cached_person, cached_clothes):
            return _fit_masks_to_source(cached_person, cached_clothes, width, height)

    if editor_has:
        return _fit_masks_to_source(editor_person, editor_clothes, width, height)

    return None, None


def parse_editor_masks(
    editor: dict | None,
) -> tuple[Image.Image | None, np.ndarray | None, np.ndarray | None]:
    """Extract RGB image and binary masks from an ImageEditor dict."""
    logger.debug("parse_editor_masks: editor=%s", _describe_value(editor))

    if not editor or not isinstance(editor, dict):
        return None, None, None

    bg_raw = editor.get("background")
    if bg_raw is None:
        comp = editor.get("composite")
        if isinstance(comp, str | dict):
            bg_raw = comp
    if bg_raw is None:
        return None, None, None

    bg_rgba = _load_background_image(bg_raw)
    if bg_rgba is None:
        return None, None, None

    bg = bg_rgba.convert("RGB")
    w, h = bg.size

    person = np.zeros((h, w), dtype=np.uint8)
    clothes = np.zeros((h, w), dtype=np.uint8)

    layers = editor.get("layers") or []
    if layers:
        layer_img = _load_layer_image(layers[0], w, h)
        if layer_img is not None:
            person, clothes = _masks_from_layer(layer_img)

    if person.sum() == 0 and clothes.sum() == 0 and editor.get("composite") is not None:
        recovered = _masks_from_composite(bg, editor["composite"], w, h)
        if recovered[0] is not None:
            person, clothes = recovered
            logger.debug(
                "parse_editor_masks: recovered from composite person=%d clothes=%d",
                int(person.sum()),
                int(clothes.sum()),
            )

    logger.debug(
        "parse_editor_masks: size=%sx%s person_pixels=%d clothes_pixels=%d",
        w,
        h,
        int(person.sum()),
        int(clothes.sum()),
    )
    return bg, person, clothes
