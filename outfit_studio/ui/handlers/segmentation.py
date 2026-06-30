"""Segmentation event handlers for GradioApp."""

from __future__ import annotations

import logging

import gradio as gr
import numpy as np
from PIL import Image

from outfit_studio.ml.segmentation_workflow import run_segmentation
from outfit_studio.ui.editor_session import (
    EditorSession,
    UploadSegmentAction,
    evaluate_upload_segment,
    resolve_clean_on_upload,
)
from outfit_studio.ui.header import SegmentationResult
from outfit_studio.ui.masks import (
    apply_masks_to_editor,
    background_key_from_image,
    background_key_from_path,
    editor_mask_reset,
    load_editor_clean_image,
    masks_have_pixels,
)
from outfit_studio.ui.operation_control import OperationCancelled, bind_request

logger = logging.getLogger(__name__)


class SegmentationHandlersMixin:
    def _run_segmentation(
        self,
        editor: dict | None,
        clean: Image.Image | None = None,
        username: str | None = None,
        debug_session_dir: str | None = None,
    ) -> SegmentationResult:
        """Run ML segmentation and build an ImageEditor value dict."""
        seg_image = load_editor_clean_image(editor) if editor else None
        if seg_image is None:
            if clean is None:
                raise ValueError("no background image")
            seg_image = clean.convert("RGB")
        else:
            seg_image = seg_image.convert("RGB")

        username = username or self.settings.default_admin
        person, clothes, active_dir = run_segmentation(
            seg_image,
            settings=self.settings,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        pipeline_clean = clean.convert("RGB") if clean is not None else seg_image
        editor_value = apply_masks_to_editor(seg_image, person, clothes, editor=editor, clean=clean)
        return SegmentationResult(editor_value, pipeline_clean, person, clothes, active_dir)

    def _try_run_segmentation(
        self,
        *,
        label: str,
        editor: dict | None,
        clean: Image.Image,
        username: str | None = None,
        debug_session_dir: str | None = None,
    ) -> SegmentationResult | None:
        try:
            return self._run_segmentation(
                editor,
                clean=clean,
                username=username,
                debug_session_dir=debug_session_dir,
            )
        except OperationCancelled:
            logger.info("%s: cancelled", label)
            return None

    @staticmethod
    def _editor_skip(
        clean_source: Image.Image | None,
        segment_key: str | None,
        suppress_hook: bool,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        return gr.skip(), clean_source, segment_key, suppress_hook, debug_session_dir

    @staticmethod
    def _segment_state_skip(
        clean_source: Image.Image | None,
        segment_key: str | None,
        suppress_hook: bool,
        debug_session_dir: str | None,
        segment_masks: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[
        None,
        Image.Image | None,
        str | None,
        bool,
        str | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        return (
            None,
            clean_source,
            segment_key,
            suppress_hook,
            debug_session_dir,
            segment_masks,
        )

    @staticmethod
    def _segment_state_result(
        editor_value: dict,
        clean: Image.Image,
        key: str,
        debug_session_dir: str | None,
        person: np.ndarray,
        clothes: np.ndarray,
    ) -> tuple[dict, Image.Image, str, bool, str | None, tuple[np.ndarray, np.ndarray]]:
        return (
            editor_value,
            clean,
            key,
            True,
            debug_session_dir,
            (person, clothes),
        )

    @staticmethod
    def _apply_pending_editor(pending: dict | None) -> dict:
        if pending is None:
            return gr.skip()
        return gr.update(value=pending)

    @staticmethod
    def _clear_pending_editor(_pending: dict | None) -> None:
        return None

    def segment(
        self,
        editor: dict | None,
        clean_source: Image.Image | None = None,
        segment_key: str | None = None,
    ) -> tuple[dict, Image.Image | None]:
        logger.info("segment: called")
        clean = self._resolve_clean_image(editor, clean_source, segment_key)
        if clean is None:
            logger.warning("segment: skipped — no background image parsed")
            return None, None
        result = self._run_segmentation(editor, clean=clean)
        return result.editor_value, result.pipeline_clean

    def prepare_upload_segment(
        self,
        editor: dict | None,
        segment_key: str | None,
        clean_source: Image.Image | None,
        suppress_upload_hook: bool,
        request: gr.Request,
        debug_session_dir: str | None,
        segment_masks: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[
        dict | None,
        Image.Image | None,
        str | None,
        bool,
        str | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        bind_request(request)
        username = self._session_username(request) or self.settings.default_admin
        debug_session_dir = self._effective_debug_dir(request, debug_session_dir)
        session = EditorSession.from_fields(
            clean_source, segment_key, debug_session_dir, suppress_upload_hook
        )
        action, clean, key = evaluate_upload_segment(editor, session)

        if action is UploadSegmentAction.SKIP_NO_BACKGROUND:
            return self._segment_state_skip(
                clean_source, segment_key, False, debug_session_dir, segment_masks
            )
        if action is UploadSegmentAction.SKIP_PROGRAMMATIC:
            return self._segment_state_skip(
                clean_source, segment_key, True, debug_session_dir, segment_masks
            )
        if action is UploadSegmentAction.SKIP_MASKS_PRESENT:
            return self._segment_state_skip(
                clean or clean_source, key, True, debug_session_dir, segment_masks
            )

        result = self._try_run_segmentation(
            label="prepare_upload_segment",
            editor=editor,
            clean=clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        if result is None:
            return self._segment_state_skip(
                clean_source, segment_key, False, debug_session_dir, segment_masks
            )
        if not masks_have_pixels(result.person, result.clothes):
            return self._segment_state_skip(
                clean_source or result.pipeline_clean,
                key,
                False,
                result.debug_session_dir,
                segment_masks,
            )

        return self._segment_state_result(
            result.editor_value,
            result.pipeline_clean,
            key,
            result.debug_session_dir,
            result.person,
            result.clothes,
        )

    def sync_clean_source(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        segment_key: str | None,
    ) -> Image.Image | None:
        session = EditorSession.from_fields(clean_source, segment_key, None, False)
        return resolve_clean_on_upload(editor, session)

    def _editor_update(
        self,
        value: dict | None,
        clean: Image.Image | None,
        key: str | None,
        debug_session_dir: str | None = None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        session = EditorSession.from_fields(None, None, debug_session_dir, False)
        editor_update = gr.update() if value is None else gr.update(value=value)
        return editor_update, *session.fields_after_programmatic_push(clean, key, debug_session_dir)

    def _segment_loaded_image(
        self,
        image: Image.Image,
        source_path: str | None = None,
        editor: dict | None = None,
        username: str | None = None,
        debug_session_dir: str | None = None,
    ) -> (
        tuple[dict, Image.Image, str, bool, str | None, tuple[np.ndarray, np.ndarray] | None] | None
    ):
        if editor is None or load_editor_clean_image(editor) is None:
            editor = {
                "background": image.convert("RGBA"),
                "layers": [],
                "composite": None,
            }
        result = self._try_run_segmentation(
            label="segment_loaded_image",
            editor=editor,
            clean=image,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        if result is None:
            return None
        key = (
            background_key_from_path(source_path)
            if source_path
            else background_key_from_image(result.pipeline_clean)
        )
        return (
            result.editor_value,
            result.pipeline_clean,
            key,
            True,
            result.debug_session_dir,
            (result.person, result.clothes),
        )

    def resegment(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        last_key: str | None,
        request: gr.Request,
        debug_session_dir: str | None,
        segment_masks: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[
        dict | None,
        Image.Image | None,
        str | None,
        bool,
        str | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        bind_request(request)
        debug_session_dir = self._effective_debug_dir(request, debug_session_dir)
        username = self._session_username(request) or self.settings.default_admin
        clean = clean_source or self._resolve_clean_image(editor, None, last_key)
        if clean is None:
            raise gr.Error("Load an image first, then click Redo Clothes Segmentation.")
        reset_editor = editor_mask_reset(editor, clean)
        result = self._try_run_segmentation(
            label="resegment",
            editor=reset_editor,
            clean=clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        if result is None:
            return self._segment_state_skip(
                clean_source, last_key, False, debug_session_dir, segment_masks
            )
        key = background_key_from_image(result.pipeline_clean) if clean is not None else last_key
        return self._segment_state_result(
            result.editor_value,
            result.pipeline_clean,
            key,
            result.debug_session_dir,
            result.person,
            result.clothes,
        )

    def segment_after_example(
        self,
        editor: dict | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        bind_request(request)
        debug_session_dir = self._effective_debug_dir(request, debug_session_dir)
        username = self._session_username(request) or self.settings.default_admin
        clean = self._resolve_clean_image(editor, None, None)
        if clean is None:
            return gr.update(), None, None, True, debug_session_dir
        result = self._try_run_segmentation(
            label="segment_after_example",
            editor=editor,
            clean=clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        if result is None:
            return self._editor_skip(None, None, True, debug_session_dir)
        key = background_key_from_image(result.pipeline_clean)
        return self._editor_update(
            result.editor_value, result.pipeline_clean, key, result.debug_session_dir
        )

    @staticmethod
    def _example_load_skip(
        clean_source: Image.Image | None,
        segment_key: str | None,
        suppress_hook: bool,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None, None]:
        return (
            gr.update(),
            clean_source,
            segment_key,
            suppress_hook,
            debug_session_dir,
            None,
        )

    def _example_load_result(
        self,
        value: dict,
        clean: Image.Image,
        key: str,
        debug_session_dir: str | None,
        masks: tuple[np.ndarray, np.ndarray],
    ) -> tuple[dict, Image.Image, str, bool, str | None, tuple[np.ndarray, np.ndarray]]:
        editor_update, *session_fields = self._editor_update(value, clean, key, debug_session_dir)
        return editor_update, *session_fields, masks

    def clear_editor_state(
        self,
    ) -> tuple[None, None, bool, None, None]:
        return (*EditorSession().cleared_fields(), None)

    @staticmethod
    def _store_example_index(evt: gr.SelectData) -> int | None:
        return evt.index if evt.selected else None

    def load_example_after_select(
        self,
        editor: dict | None,
        index: int | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[
        dict,
        Image.Image | None,
        str | None,
        bool,
        str | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        bind_request(request)
        debug_session_dir = self._effective_debug_dir(request, debug_session_dir)
        username = self._session_username(request) or self.settings.default_admin
        path: str | None = None
        if index is not None and 0 <= index < len(self.examples):
            path = self.examples[index]
        if path:
            image = self._open_image(path)
            if image is not None:
                loaded = self._segment_loaded_image(
                    image,
                    source_path=path,
                    editor=editor,
                    username=username,
                    debug_session_dir=debug_session_dir,
                )
                if loaded is None:
                    return self._example_load_skip(None, None, True, debug_session_dir)
                value, clean, key, _, new_debug_dir, masks = loaded
                return self._example_load_result(value, clean, key, new_debug_dir, masks)
        seg_result = self.segment_after_example(editor, request, debug_session_dir)
        return (*seg_result, None)

    def use_result_as_input(
        self,
        slider_val: tuple | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[
        dict,
        Image.Image | None,
        str | None,
        bool,
        str | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        bind_request(request)
        if not slider_val:
            return self._example_load_skip(None, None, False, debug_session_dir)
        debug_session_dir = self._effective_debug_dir(request, debug_session_dir)
        username = self._session_username(request) or self.settings.default_admin
        _, after = slider_val
        clean = after.convert("RGB")
        loaded = self._segment_loaded_image(
            clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        if loaded is None:
            return self._example_load_skip(None, None, False, debug_session_dir)
        value, clean, key, _, new_debug_dir, masks = loaded
        return self._example_load_result(value, clean, key, new_debug_dir, masks)
