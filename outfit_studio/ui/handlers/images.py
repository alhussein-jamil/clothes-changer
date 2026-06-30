"""Image loading and example discovery for GradioApp."""

from __future__ import annotations

import logging
from pathlib import Path

from gradio_client import handle_file
from PIL import Image

from outfit_studio.ui.masks import (
    file_path_from_editor,
    image_from_segment_key,
    load_editor_clean_image,
)
from outfit_studio.ui.theme import UI
from outfit_studio.utils.image import resize_max

logger = logging.getLogger(__name__)


class ImageHandlersMixin:
    def _load_examples(self) -> list[str]:
        candidate = self.settings.resolved_examples_dir
        if candidate.is_dir():
            files = sorted(
                str(p.resolve())
                for p in candidate.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
            if files:
                logger.info(
                    "Loaded %d example images from %s",
                    min(len(files), UI.MAX_EXAMPLES),
                    candidate,
                )
                return files[: UI.MAX_EXAMPLES]
        logger.debug("No example images found")
        return []

    def _resolve_clean_image(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        segment_key: str | None = None,
    ) -> Image.Image | None:
        """Best-effort source RGB for segmentation when the editor payload is incomplete."""
        if clean_source is not None:
            return clean_source.convert("RGB")
        clean = load_editor_clean_image(editor)
        if clean is not None:
            return clean
        return image_from_segment_key(segment_key)

    def _pipeline_source(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        segment_key: str | None,
    ) -> Image.Image | None:
        """Unmasked photo for inpainting — never the editor composite overlay."""
        if clean_source is not None:
            return clean_source.convert("RGB")
        from_path = image_from_segment_key(segment_key)
        if from_path is not None:
            return from_path
        return load_editor_clean_image(editor)

    def _open_image(self, source: Image.Image | str | Path | dict) -> Image.Image | None:
        src_label = (
            type(source).__name__ if not isinstance(source, str) else source[: UI.LOG_PREVIEW_LEN]
        )
        logger.info("open_image: source=%s", src_label)
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        elif isinstance(source, dict):
            path = file_path_from_editor(source)
            if path is None:
                logger.warning("open_image: no file path in editor dict")
                return None
            source = path
            resolved = handle_file(str(source))
            if isinstance(resolved, dict):
                resolved = resolved.get("path")
            if not resolved:
                logger.warning("open_image: handle_file returned nothing for %r", source)
                return None
            path = Path(resolved)
            if not path.is_file():
                logger.warning("open_image: file does not exist: %s", path)
                return None
            image = Image.open(path).convert("RGB")
            logger.info("open_image: loaded %s (%sx%s)", path, image.width, image.height)
        else:
            resolved = handle_file(str(source))
            if isinstance(resolved, dict):
                resolved = resolved.get("path")
            if not resolved:
                logger.warning("open_image: handle_file returned nothing for %r", source)
                return None
            path = Path(resolved)
            if not path.is_file():
                logger.warning("open_image: file does not exist: %s", path)
                return None
            image = Image.open(path).convert("RGB")
            logger.info("open_image: loaded %s (%sx%s)", path, image.width, image.height)

        max_size = self.settings.max_image_size
        if max(image.size) > max_size:
            image = resize_max(image, max_size)
            logger.info(
                "open_image: resized to %sx%s (max=%d)",
                image.width,
                image.height,
                max_size,
            )
        return image

    def _path_from_select(self, value: object, index: int | None = None) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):
            if "background" in value:
                return file_path_from_editor(value)
            for key in ("path", "name"):
                if value.get(key):
                    return str(value[key])
            image = value.get("image")
            if isinstance(image, dict):
                for key in ("path", "name"):
                    if image.get(key):
                        return str(image[key])
            return file_path_from_editor(value)
        if isinstance(value, list | tuple):
            if len(value) == 2 and isinstance(value[1], str) and isinstance(value[0], str):
                candidate = value[0]
                if Path(candidate).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    return candidate
            if len(value) == 2:
                sample = value[1]
                if isinstance(sample, dict):
                    return self._path_from_select(sample)
                if isinstance(sample, list | tuple):
                    if not sample:
                        return None
                    first = sample[0]
                    if isinstance(first, dict):
                        return self._path_from_select(first)
                    return str(first)
                if isinstance(sample, str):
                    return sample
            first = value[0]
            if isinstance(first, dict):
                return self._path_from_select(first)
            if isinstance(first, str):
                return first
        if isinstance(value, str):
            return value
        if index is not None and 0 <= index < len(self.examples):
            return self.examples[index]
        return None
