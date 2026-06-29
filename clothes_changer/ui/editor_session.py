"""Editor session state and upload-segmentation policy for the Gradio ImageEditor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PIL import Image

from clothes_changer.ui.masks import (
    background_key_from_image,
    load_editor_clean_image,
    masks_have_pixels,
    parse_editor_masks,
)

# Gradio ``gr.State`` tuple: clean_source, segment_key, suppress_upload_hook, debug_session_dir
SessionFields = tuple[Image.Image | None, str | None, bool, str | None]


class UploadSegmentAction(str, Enum):
    """Whether an ImageEditor upload should trigger auto-segmentation."""

    SEGMENT = "segment"
    SKIP_NO_BACKGROUND = "skip_no_background"
    SKIP_PROGRAMMATIC = "skip_programmatic"
    SKIP_MASKS_PRESENT = "skip_masks_present"


@dataclass(frozen=True)
class EditorSession:
    """Shadow state kept alongside the ImageEditor in Gradio ``gr.State`` components."""

    clean_source: Image.Image | None = None
    segment_key: str | None = None
    debug_session_dir: str | None = None
    suppress_upload_hook: bool = False

    @classmethod
    def from_fields(
        cls,
        clean_source: Image.Image | None,
        segment_key: str | None,
        debug_session_dir: str | None,
        suppress_upload_hook: bool,
    ) -> EditorSession:
        return cls(clean_source, segment_key, debug_session_dir, suppress_upload_hook)

    def cleared_fields(self) -> SessionFields:
        """Reset all session fields (ImageEditor clear event)."""
        return None, None, False, None

    def fields_after_programmatic_push(
        self,
        clean_source: Image.Image | None = None,
        segment_key: str | None = None,
        debug_session_dir: str | None = None,
    ) -> SessionFields:
        """State after pushing masks into the editor without a user upload."""
        return (
            clean_source if clean_source is not None else self.clean_source,
            segment_key if segment_key is not None else self.segment_key,
            True,
            debug_session_dir if debug_session_dir is not None else self.debug_session_dir,
        )

    def fields_after_segmentation(
        self,
        clean: Image.Image,
        key: str,
        debug_session_dir: str | None,
    ) -> SessionFields:
        """State after auto-segmentation writes masks into the editor."""
        return clean, key, True, debug_session_dir

    def with_clean_source(self, clean_source: Image.Image | None) -> EditorSession:
        return EditorSession(
            clean_source,
            self.segment_key,
            self.debug_session_dir,
            self.suppress_upload_hook,
        )


def resolve_clean_on_upload(
    editor: dict | None,
    session: EditorSession,
) -> Image.Image | None:
    """Resolve pristine RGB on upload; never return a mask-tinted composite."""
    clean = session.clean_source
    key = session.segment_key
    if clean is not None:
        if key is None:
            return clean
        resolved = load_editor_clean_image(editor)
        if resolved is None:
            return clean
        if background_key_from_image(resolved.convert("RGB")) == key:
            return clean
        return resolved.convert("RGB")
    return load_editor_clean_image(editor)


def evaluate_upload_segment(
    editor: dict | None,
    session: EditorSession,
) -> tuple[UploadSegmentAction, Image.Image | None, str | None]:
    """Decide whether an upload should auto-segment and derive the image fingerprint."""
    bg, person, clothes = parse_editor_masks(editor)
    if bg is None:
        return UploadSegmentAction.SKIP_NO_BACKGROUND, session.clean_source, session.segment_key

    clean = load_editor_clean_image(editor) or bg.convert("RGB")
    key = background_key_from_image(clean)

    if (
        session.suppress_upload_hook
        and session.segment_key is not None
        and key == session.segment_key
    ):
        return UploadSegmentAction.SKIP_PROGRAMMATIC, session.clean_source, session.segment_key

    layers = (editor or {}).get("layers") or []
    if (
        len(layers) > 0
        and masks_have_pixels(person, clothes)
        and session.segment_key is not None
        and key == session.segment_key
    ):
        return UploadSegmentAction.SKIP_MASKS_PRESENT, session.clean_source or clean, key

    return UploadSegmentAction.SEGMENT, clean, key
