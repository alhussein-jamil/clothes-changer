"""Gradio web interface."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import NamedTuple

import gradio as gr
import numpy as np
from gradio_client import handle_file
from gradio_imageslider import ImageSlider
from PIL import Image

from clothes_changer.config import PROJECT_ROOT, Settings, get_settings
from clothes_changer.content_config import (
    get_app_name,
    get_default_negative_prompt,
    get_default_prompt,
    get_title_html,
)
from clothes_changer.db.database import Database
from clothes_changer.ml.inpainter import get_inpaint_engine
from clothes_changer.ml.pipeline import get_pipeline
from clothes_changer.ui.constants import CLOTHES_COLOR, CUSTOM_CSS, EDITOR_CANVAS_SIZE, PERSON_COLOR, SEED_MAX
from clothes_changer.ui.masks import (
    apply_masks_to_editor,
    background_key_from_image,
    background_key_from_path,
    editor_mask_reset,
    file_path_from_editor,
    image_from_segment_key,
    load_editor_clean_image,
    masks_have_pixels,
    parse_editor_masks,
)
from clothes_changer.utils.image import align_masks

logger = logging.getLogger(__name__)


class SegmentationResult(NamedTuple):
    """Return value from ``_run_segmentation``."""

    editor_value: dict
    pipeline_clean: Image.Image
    person: np.ndarray
    clothes: np.ndarray
    debug_session_dir: str | None


class GradioApp:
    """Clothes Changer Gradio UI."""

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or Database()
        self.pipeline = get_pipeline()
        self.examples = self._load_examples()
        self._refresh_models()
        logger.info(
            "GradioApp initialized (%d examples, %d models)",
            len(self.examples),
            len(self.model_ids),
        )

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

        from clothes_changer.ml.gpu_memory import release_inpaint_gpu, release_segmentation_gpu
        from clothes_changer.ml.pipeline_debug import PipelineDebugSession
        from clothes_changer.ml.segmentor import get_segmentor

        username = username or self.settings.default_admin
        session, active_dir = PipelineDebugSession.open_or_create(
            self.settings, username, debug_session_dir
        )
        seg_debug = None
        if session is not None:
            seg_debug = session.subfolder("segmentation")
            seg_debug.metadata.update(
                {
                    "username": username,
                    "segformer_model": self.settings.segformer_model,
                    "u2net_model": self.settings.extra_clothes_model,
                    "image_size": list(seg_image.size),
                }
            )
            seg_debug.save_image("00_source.png", seg_image)

        release_inpaint_gpu()
        logger.info(
            "segment: running segmentor on %sx%s image (user=%r)",
            seg_image.width,
            seg_image.height,
            username,
        )
        _, person, clothes = get_segmentor().segment(seg_image, debug=seg_debug)
        release_segmentation_gpu()
        logger.info(
            "segment: done — person_pixels=%d clothes_pixels=%d",
            int(person.sum()),
            int(clothes.sum()),
        )
        pipeline_clean = clean.convert("RGB") if clean is not None else seg_image
        editor_value = apply_masks_to_editor(seg_image, person, clothes, editor=editor, clean=clean)
        return SegmentationResult(editor_value, pipeline_clean, person, clothes, active_dir)

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
        last_key: str | None,
        clean_source: Image.Image | None,
        skip_upload: bool,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Segment on upload and push masks straight into the ImageEditor."""
        username = self._session_username(request) or self.settings.default_admin
        if skip_upload:
            logger.info("prepare_upload_segment: skipped — programmatic update")
            return gr.skip(), clean_source, last_key, False, debug_session_dir

        bg, person, clothes = parse_editor_masks(editor)
        if bg is None:
            logger.warning("prepare_upload_segment: no editor background yet")
            return gr.skip(), clean_source, last_key, False, debug_session_dir

        clean = load_editor_clean_image(editor) or bg.convert("RGB")
        key = background_key_from_image(clean)
        layers = (editor or {}).get("layers") or []
        if len(layers) > 0 and masks_have_pixels(person, clothes) and key == last_key:
            logger.info("prepare_upload_segment: skipped — masks already on editor")
            return gr.skip(), clean_source or clean, key, False, debug_session_dir

        logger.info(
            "prepare_upload_segment: running segmentation on %sx%s image",
            clean.width,
            clean.height,
        )
        result = self._run_segmentation(
            editor,
            clean=clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        if not masks_have_pixels(result.person, result.clothes):
            logger.warning("prepare_upload_segment: empty segment output for %s", key)
            return (
                gr.skip(),
                clean_source or result.pipeline_clean,
                key,
                False,
                result.debug_session_dir,
            )

        return (
            gr.update(value=result.editor_value),
            result.pipeline_clean,
            key,
            True,
            result.debug_session_dir,
        )

    def sync_clean_source(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        segment_key: str | None,
    ) -> Image.Image | None:
        """Preserve the pristine upload; editor payloads may bake mask colors into pixels."""
        if clean_source is not None:
            return clean_source
        return self._resolve_clean_image(editor, None, segment_key)

    def _load_examples(self) -> list[str]:
        for candidate in (
            self.settings.resolved_examples_dir,
            PROJECT_ROOT / "examples",
            PROJECT_ROOT.parent / "Clothless" / "examples",
            PROJECT_ROOT.parent / "examples",
        ):
            if candidate.is_dir():
                files = sorted(
                    str(p.resolve())
                    for p in candidate.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                )
                if files:
                    logger.info("Loaded %d example images from %s", min(len(files), 12), candidate)
                    return files[:12]
        logger.debug("No example images found")
        return []

    def example_gallery_items(self) -> list[tuple[str, str]]:
        """Gallery rows for sample inputs on the Generate tab."""
        return [(path, Path(path).name) for path in self.examples]

    def _refresh_models(self) -> None:
        engine = get_inpaint_engine()
        models = engine.list_models()
        self.models = models
        self.model_ids = [m["id"] for m in models]
        self.model_choices = [(f"{m['name']} ({m['arch']})", m["id"]) for m in models]
        self.default_model = engine.default_model_id()
        logger.debug(
            "Model list refreshed — default=%s choices=%d",
            self.default_model,
            len(self.model_ids),
        )

    def authenticate(self, username: str, password: str) -> bool:
        ok = self.db.authenticate(username, password)
        if ok:
            logger.info("User %r logged in", username)
        return ok

    def _session_username(self, request: gr.Request | None) -> str | None:
        """Logged-in user, or default admin when auth is disabled."""
        name = getattr(request, "username", None) if request is not None else None
        if name:
            return name
        if not self.settings.require_auth:
            return self.settings.default_admin
        return None

    def _user_label(self, request: gr.Request) -> str:
        name = self._session_username(request) or "Guest"
        return f"Welcome, {name}"

    def _credits_label(self, request: gr.Request) -> str:
        username = self._session_username(request)
        if not username:
            return "0 credits"
        user = self.db.get_user(username)
        if not user:
            return "0 credits"
        if user.is_admin:
            return "Unlimited credits (admin)"
        return f"{user.credits} credits"

    def _open_image(self, source: Image.Image | str | Path | dict) -> Image.Image | None:
        src_label = type(source).__name__ if not isinstance(source, str) else source[:120]
        logger.info("open_image: source=%s", src_label)
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, dict):
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
            # Gallery tuple: (path, caption) or nested FileData
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

    def _editor_from_select(self, value: object) -> dict | None:
        if isinstance(value, dict) and "background" in value:
            return value
        if isinstance(value, list | tuple) and len(value) == 2:
            sample = value[1]
            if isinstance(sample, dict) and "background" in sample:
                return sample
            if isinstance(sample, list | tuple) and sample and isinstance(sample[0], dict):
                return sample[0]
        return None

    def _editor_update(
        self,
        value: dict | None,
        clean: Image.Image | None,
        key: str | None,
        debug_session_dir: str | None = None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Return a Gradio ImageEditor update and suppress the upload hook."""
        if value is None:
            return gr.update(), clean, key, True, debug_session_dir
        return gr.update(value=value), clean, key, True, debug_session_dir

    def _segment_loaded_image(
        self,
        image: Image.Image,
        source_path: str | None = None,
        editor: dict | None = None,
        username: str | None = None,
        debug_session_dir: str | None = None,
    ) -> tuple[dict, Image.Image, str, bool, str | None]:
        """Segment a file-backed or gallery image; suppress the follow-up upload hook."""
        if editor is None or load_editor_clean_image(editor) is None:
            editor = {
                "background": image.convert("RGBA"),
                "layers": [],
                "composite": None,
            }
        result = self._run_segmentation(
            editor,
            clean=image,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        key = (
            background_key_from_path(source_path)
            if source_path
            else background_key_from_image(result.pipeline_clean)
        )
        return result.editor_value, result.pipeline_clean, key, True, result.debug_session_dir

    def segment_if_unmasked(self, editor: dict | None) -> tuple[dict, Image.Image | None]:
        """Auto-segment when an image is present but mask layers are empty."""
        logger.info("segment_if_unmasked: called")
        bg, person, clothes = parse_editor_masks(editor)
        if bg is None:
            logger.warning("segment_if_unmasked: skipped — background is empty")
            return gr.update(), None
        if person is not None and clothes is not None and (person.sum() > 0 or clothes.sum() > 0):
            logger.info(
                "segment_if_unmasked: skipped — masks already present (person=%d clothes=%d)",
                int(person.sum()),
                int(clothes.sum()),
            )
            return gr.update(), load_editor_clean_image(editor) or bg.convert("RGB")
        logger.info("segment_if_unmasked: no masks yet, running segment")
        value, clean = self.segment(editor)
        if value is None:
            return gr.update(), None
        return gr.update(value=value), clean

    def resegment_prepare(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        last_key: str | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Clear mask layers before re-segmenting (Gradio appends layers otherwise)."""
        logger.info("resegment: clearing existing mask layers")
        clean = self._resolve_clean_image(editor, clean_source, last_key)
        if clean is None:
            raise gr.Error("Load an image first, then click Redo Clothes Segmentation.")
        reset = editor_mask_reset(editor, clean)
        return gr.update(value=reset), clean, last_key, True, debug_session_dir

    def resegment(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        last_key: str | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Force re-segmentation (Redo button) — replaces mask layer, never stacks."""
        logger.info("resegment: replacing existing mask layer")
        username = self._session_username(request) or self.settings.default_admin
        clean = clean_source or self._resolve_clean_image(editor, None, last_key)
        if clean is None:
            raise gr.Error("Load an image first, then click Redo Clothes Segmentation.")
        result = self._run_segmentation(
            editor,
            clean=clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        key = background_key_from_image(result.pipeline_clean) if clean is not None else last_key
        return self._editor_update(
            result.editor_value, result.pipeline_clean, key, result.debug_session_dir
        )

    def segment_after_example(
        self,
        editor: dict | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Segment after gr.Examples loads into the editor (canvas already fitted)."""
        logger.info("segment_after_example: called")
        username = self._session_username(request) or self.settings.default_admin
        clean = self._resolve_clean_image(editor, None, None)
        if clean is None:
            return gr.update(), None, None, True, debug_session_dir
        result = self._run_segmentation(
            editor,
            clean=clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        key = background_key_from_image(result.pipeline_clean)
        return self._editor_update(
            result.editor_value, result.pipeline_clean, key, result.debug_session_dir
        )

    def clear_editor_state(self) -> tuple[None, None, None]:
        return None, None, None

    def select_image_and_segment(
        self,
        evt: gr.SelectData,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Load an image from history gallery and run segmentation."""
        username = self._session_username(request) or self.settings.default_admin
        logger.info(
            "select_image_and_segment: selected=%s index=%s value=%r",
            evt.selected,
            evt.index,
            evt.value,
        )
        if not evt.selected:
            return gr.update(), None, None, False, debug_session_dir
        path = self._path_from_select(evt.value, evt.index)
        if not path:
            items = self.history_images(request)
            if evt.index is not None and 0 <= evt.index < len(items):
                path = items[evt.index][0]
        if not path:
            logger.warning("select_image_and_segment: could not parse path from %r", evt.value)
            return gr.update(), None, None, False, debug_session_dir
        image = self._open_image(path)
        if image is None:
            return gr.update(), None, None, False, debug_session_dir
        value, clean, key, _, new_debug_dir = self._segment_loaded_image(
            image,
            source_path=path,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        return self._editor_update(value, clean, key, new_debug_dir)

    def load_history_and_segment(
        self,
        evt: gr.SelectData,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Load a history gallery image and run segmentation."""
        return self.select_image_and_segment(evt, request, debug_session_dir)

    @staticmethod
    def _store_example_index(evt: gr.SelectData) -> int | None:
        return evt.index if evt.selected else None

    def load_example_after_select(
        self,
        editor: dict | None,
        index: int | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        """Segment after gr.Examples has populated the ImageEditor."""
        logger.info("load_example_after_select: index=%s", index)
        username = self._session_username(request) or self.settings.default_admin
        path: str | None = None
        if index is not None and 0 <= index < len(self.examples):
            path = self.examples[index]
        if path:
            image = self._open_image(path)
            if image is not None:
                value, clean, key, _, new_debug_dir = self._segment_loaded_image(
                    image,
                    source_path=path,
                    editor=editor,
                    username=username,
                    debug_session_dir=debug_session_dir,
                )
                return self._editor_update(value, clean, key, new_debug_dir)
        return self.segment_after_example(editor, request, debug_session_dir)

    def use_result_as_input(
        self,
        slider_val: tuple | None,
        request: gr.Request,
        debug_session_dir: str | None,
    ) -> tuple[dict, Image.Image | None, str | None, bool, str | None]:
        if not slider_val:
            return gr.update(), None, None, False, debug_session_dir
        username = self._session_username(request) or self.settings.default_admin
        _, after = slider_val
        clean = after.convert("RGB")
        value, clean, key, _, new_debug_dir = self._segment_loaded_image(
            clean,
            username=username,
            debug_session_dir=debug_session_dir,
        )
        return self._editor_update(value, clean, key, new_debug_dir)

    def generate(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        segment_key: str | None,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        use_controlnet: bool,
        steps: int,
        guidance_scale: float,
        seed: int,
        random_seed: bool,
        debug_session_dir: str | None,
        request: gr.Request,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[tuple[Image.Image, Image.Image] | None, int, str | None]:
        username = self._session_username(request)
        logger.info(
            "generate: user=%r model=%r steps=%d cfg=%.1f controlnet=%s random_seed=%s",
            username,
            model_id,
            steps,
            guidance_scale,
            use_controlnet,
            random_seed,
        )
        if not username:
            raise gr.Error("Not authenticated")

        user = self.db.get_user(username)
        if not user:
            raise gr.Error("User not found")
        if not user.is_admin and user.credits <= 0:
            raise gr.Error("No credits remaining. Contact an administrator.")

        if not prompt or not prompt.strip():
            raise gr.Error("Prompt cannot be empty")

        progress(0, desc="Preparing generation")

        source, person_mask, clothes_mask = parse_editor_masks(editor)
        pipeline_image = self._pipeline_source(editor, clean_source, segment_key)
        if pipeline_image is None:
            return None, seed, debug_session_dir
        source = pipeline_image

        if (
            person_mask is not None
            and clothes_mask is not None
            and person_mask.shape != (source.height, source.width)
        ):
            person_mask, clothes_mask = align_masks(
                person_mask, clothes_mask, source.height, source.width
            )

        if (
            person_mask is None
            or clothes_mask is None
            or (person_mask.sum() == 0 and clothes_mask.sum() == 0)
        ):
            logger.info("generate: masks empty — running on-the-fly segmentation")
            from clothes_changer.ml.gpu_memory import release_inpaint_gpu
            from clothes_changer.ml.pipeline_debug import PipelineDebugSession
            from clothes_changer.ml.segmentor import get_segmentor

            progress(0.05, desc="Running clothes segmentation")
            release_inpaint_gpu()
            session, active_dir = PipelineDebugSession.open_or_create(
                self.settings, username, debug_session_dir
            )
            seg_debug = session.subfolder("segmentation") if session else None
            _, person_mask, clothes_mask = get_segmentor().segment(source, debug=seg_debug)
            debug_session_dir = active_dir

        model_id = model_id if model_id in self.model_ids else self.default_model
        actual_seed = random.randint(0, SEED_MAX) if random_seed else int(seed)
        logger.info("generate: resolved model=%s seed=%d", model_id, actual_seed)

        def report_progress(fraction: float, desc: str) -> None:
            progress(fraction, desc=desc)

        try:
            result, filename, active_debug_dir = self.pipeline.generate(
                image=source,
                person_mask=person_mask,
                clothes_mask=clothes_mask,
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip(),
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                seed=actual_seed,
                model=model_id,
                use_controlnet=use_controlnet,
                username=username,
                progress=report_progress,
                debug_session_dir=debug_session_dir,
            )
        except Exception as e:
            logger.exception("Generation failed")
            raise gr.Error(str(e)) from e

        if not user.is_admin:
            self.db.deduct_credit(username)

        full_prompt = f"+: {prompt} | -: {negative_prompt}"
        self.db.log_image(username, filename, full_prompt)
        logger.info("generate: success → %s", filename)

        return gr.update(value=(source, result.convert("RGB"))), actual_seed, active_debug_dir or debug_session_dir

    @staticmethod
    def _debug_status_update(debug_session_dir: str | None) -> gr.Markdown:
        settings = get_settings()
        if not settings.pipeline_debug:
            return gr.update(visible=False)
        if debug_session_dir:
            return gr.update(
                value=f"**Debug artifacts:** `{debug_session_dir}`",
                visible=True,
            )
        return gr.update(
            value="**Debug mode:** waiting for segmentation or generation…",
            visible=True,
        )

    def history_images(self, request: gr.Request | None = None) -> list[tuple[str, str]]:
        """Return gallery items as (absolute_path, caption) tuples."""
        username = self._session_username(request)
        if not username and not self.settings.require_auth:
            username = self.settings.default_admin
        if not username:
            return []

        out_dir = self.settings.resolved_output_dir
        seen: set[str] = set()
        items: list[tuple[str, str]] = []

        for row in self.db.get_history(username):
            fn = row.get("filename")
            if not fn or fn in seen:
                continue
            path = out_dir / fn
            if path.is_file():
                seen.add(fn)
                caption = (row.get("prompt") or fn)[:80]
                items.append((str(path.resolve()), caption))

        # Also pick up files on disk that may not be in DB yet
        for path in sorted(out_dir.glob(f"{username}_*.png"), reverse=True):
            if path.name in seen:
                continue
            items.append((str(path.resolve()), path.name))
            seen.add(path.name)

        logger.debug("history_images: %d items for %r", len(items), username)
        return items[:48]

    def is_admin(self, request: gr.Request) -> bool:
        username = self._session_username(request)
        if not username:
            return False
        user = self.db.get_user(username)
        return bool(user and user.is_admin)

    def list_users_table(self) -> list[list]:
        return [
            [u.id, u.username, u.credits, "Yes" if u.is_admin else "No"]
            for u in self.db.list_users()
        ]

    def update_credits(self, username: str, credits: int) -> str:
        if not username or credits < 0:
            logger.warning("update_credits: invalid input user=%r credits=%r", username, credits)
            return "Invalid input"
        result = "Updated" if self.db.set_credits(username, int(credits)) else "Failed"
        logger.info("update_credits: %s for %r → %d", result, username, credits)
        return result

    def _admin_panel_boot(self, request: gr.Request) -> tuple[dict, list[list] | dict]:
        """Show admin accordion and populate users table for admins only."""
        if self.is_admin(request):
            return gr.update(visible=True), self.list_users_table()
        return gr.update(visible=False), gr.skip()

    def _load_history_on_tab(self, evt: gr.SelectData, request: gr.Request) -> list | dict:
        """Load gallery when the History tab is opened."""
        if evt.selected and evt.index == 1:
            return self.history_images(request)
        return gr.update()

    def history_gallery_value(self, request: gr.Request | None = None) -> list[tuple[str, str]]:
        """Periodic / on-load refresh for the history gallery."""
        return self.history_images(request)

    def create_ui(self) -> gr.Blocks:
        with gr.Blocks(css=CUSTOM_CSS, title=get_app_name()) as demo:
            gr.HTML(get_title_html())

            with gr.Tabs() as main_tabs:
                with gr.Tab("Generate"):
                    with gr.Row():
                        user_info = gr.Textbox(show_label=False, interactive=False)
                        credits_info = gr.Textbox(show_label=False, interactive=False)

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            input_image = gr.ImageEditor(
                                label="Input Image",
                                type="pil",
                                format="png",
                                interactive=True,
                                image_mode="RGBA",
                                layers=False,
                                canvas_size=EDITOR_CANVAS_SIZE,
                                brush=gr.Brush(
                                    colors=[f"rgba{PERSON_COLOR}", f"rgba{CLOTHES_COLOR}"],
                                    color_mode="fixed",
                                    default_color=f"rgba{CLOTHES_COLOR}",
                                ),
                            )
                            resegment_btn = gr.Button("Redo Clothes Segmentation", size="sm")
                            clean_source = gr.State(value=None)
                            segment_key = gr.State(value=None)
                            skip_upload_segment = gr.State(value=False)
                            debug_session_dir = gr.State(value=None)
                            if self.settings.pipeline_debug:
                                debug_status = gr.Markdown(
                                    value="**Debug mode:** artifacts saved under `debug-pipeline/`",
                                )
                            else:
                                debug_status = gr.Markdown(visible=False)

                        with gr.Column(scale=1):
                            result = ImageSlider(
                                label="Before / After",
                                interactive=False,
                            )
                            use_as_input = gr.Button("Use as Input", visible=False, size="sm")
                            generate_btn = gr.Button("Generate", variant="primary", size="lg")
                            if self.examples:
                                examples = gr.Examples(
                                    examples=self.examples,
                                    inputs=input_image,
                                    label="Examples",
                                )
                            else:
                                examples = None
                            example_index = gr.State(value=None)

                    with gr.Accordion("Model & prompts", open=True):
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=self.model_choices,
                                value=self.default_model,
                                label="Model",
                            )
                            use_controlnet = gr.Checkbox(
                                label="Pose ControlNet",
                                value=self.settings.use_controlnet,
                            )
                            reload_btn = gr.Button("↻ Models")
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            value=get_default_prompt(),
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            lines=2,
                            value=get_default_negative_prompt(),
                        )

                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            steps = gr.Slider(
                                10, 100, value=self.settings.inpaint_steps, step=1, label="Steps"
                            )
                            guidance = gr.Slider(
                                1, 20, value=self.settings.guidance_scale, step=0.5, label="CFG"
                            )
                            seed = gr.Slider(0, SEED_MAX, value=42, step=1, label="Seed")
                            random_seed = gr.Checkbox(label="Random seed", value=True)

                with gr.Tab("History"):
                    history_gallery = gr.Gallery(
                        value=self.history_gallery_value,
                        every=10,
                        label="Your generations — click to reload into the editor",
                        columns=4,
                        height=420,
                        object_fit="contain",
                        allow_preview=True,
                        show_download_button=True,
                    )
                    history_refresh = gr.Button("Refresh history")

            with gr.Accordion("Admin", visible=False, open=False) as admin_panel:
                gr.Markdown("### User management")
                users_df = gr.Dataframe(
                    headers=["id", "username", "credits", "admin"],
                    interactive=False,
                )
                gr.Button("Refresh").click(self.list_users_table, None, users_df)
                with gr.Row():
                    admin_user = gr.Textbox(label="Username")
                    admin_credits = gr.Number(label="Credits", value=10, minimum=0)
                    admin_set = gr.Button("Set credits")
                admin_msg = gr.Textbox(label="Result", interactive=False)
                admin_set.click(self.update_credits, [admin_user, admin_credits], admin_msg)

            # Event wiring (Admin is not a hidden Tab — breaks Gradio 5 mount as Tab)
            demo.load(self._admin_panel_boot, None, [admin_panel, users_df])
            demo.load(self._user_label, None, user_info)
            demo.load(self._credits_label, None, credits_info)
            demo.load(self.history_gallery_value, None, history_gallery)
            main_tabs.select(self._load_history_on_tab, None, history_gallery)

            history_refresh.click(self.history_images, None, history_gallery)

            input_image.upload(
                self.sync_clean_source,
                inputs=[input_image, clean_source, segment_key],
                outputs=clean_source,
            ).success(
                self.prepare_upload_segment,
                inputs=[
                    input_image,
                    segment_key,
                    clean_source,
                    skip_upload_segment,
                    debug_session_dir,
                ],
                outputs=[
                    input_image,
                    clean_source,
                    segment_key,
                    skip_upload_segment,
                    debug_session_dir,
                ],
            )
            input_image.clear(
                self.clear_editor_state,
                None,
                [clean_source, segment_key, debug_session_dir],
            )
            resegment_btn.click(
                self.resegment_prepare,
                [input_image, clean_source, segment_key, debug_session_dir],
                [
                    input_image,
                    clean_source,
                    segment_key,
                    skip_upload_segment,
                    debug_session_dir,
                ],
            ).then(
                self.resegment,
                [input_image, clean_source, segment_key, debug_session_dir],
                [
                    input_image,
                    clean_source,
                    segment_key,
                    skip_upload_segment,
                    debug_session_dir,
                ],
            ).then(
                self._debug_status_update,
                debug_session_dir,
                debug_status,
            )

            def reload_models() -> gr.Dropdown:
                self._refresh_models()
                return gr.Dropdown(
                    choices=self.model_choices,
                    value=self.default_model,
                )

            reload_btn.click(reload_models, None, model_dropdown)

            random_seed.change(
                lambda r: gr.update(interactive=not r),
                random_seed,
                seed,
            )

            generate_btn.click(
                lambda: gr.update(value=None),
                None,
                result,
            ).then(
                self.generate,
                [
                    input_image,
                    clean_source,
                    segment_key,
                    prompt,
                    negative_prompt,
                    model_dropdown,
                    use_controlnet,
                    steps,
                    guidance,
                    seed,
                    random_seed,
                    debug_session_dir,
                ],
                [result, seed, debug_session_dir],
            ).then(
                self._debug_status_update,
                debug_session_dir,
                debug_status,
            ).then(lambda: gr.update(visible=True), None, use_as_input).then(
                self._credits_label, None, credits_info
            ).then(self.history_images, None, history_gallery)

            use_as_input.click(
                self.use_result_as_input,
                [result, debug_session_dir],
                [input_image, clean_source, segment_key, skip_upload_segment, debug_session_dir],
            )

            history_gallery.select(
                self.load_history_and_segment,
                debug_session_dir,
                [input_image, clean_source, segment_key, skip_upload_segment, debug_session_dir],
            )

            if examples is not None:
                examples.dataset.select(
                    self._store_example_index,
                    None,
                    example_index,
                ).then(
                    self.load_example_after_select,
                    inputs=[input_image, example_index, debug_session_dir],
                    outputs=[
                        input_image,
                        clean_source,
                        segment_key,
                        skip_upload_segment,
                        debug_session_dir,
                    ],
                ).then(
                    self._debug_status_update,
                    debug_session_dir,
                    debug_status,
                )

        return demo

    def _allowed_paths(self) -> list[str]:
        paths = {
            self.settings.resolved_output_dir,
            self.settings.resolved_models_dir,
            self.settings.resolved_examples_dir,
            PROJECT_ROOT / "examples",
            PROJECT_ROOT.parent / "Clothless" / "examples",
            PROJECT_ROOT.parent / "examples",
        }
        return [str(p.resolve()) for p in paths]

    def launch(self) -> None:
        self.settings.ensure_dirs()
        if not self.db.user_exists(self.settings.default_admin):
            try:
                self.db.register_user(
                    self.settings.default_admin,
                    self.settings.default_password,
                    credits=self.settings.default_credits,
                    is_admin=True,
                )
                logger.info(
                    "Bootstrapped default admin %r",
                    self.settings.default_admin,
                )
            except Exception as e:
                logger.warning("Admin bootstrap: %s", e)

        logger.info("Building Gradio UI …")
        demo = self.create_ui()
        favicon = (
            self.settings.resolved_favicon_path
            if self.settings.resolved_favicon_path.is_file()
            else None
        )
        allowed = self._allowed_paths()
        logger.info(
            "Starting server %s:%d (share=%s, auth=%s, %d allowed paths)",
            self.settings.host,
            self.settings.port,
            self.settings.enable_sharing,
            self.settings.require_auth,
            len(allowed),
        )
        launch_kwargs: dict = {
            "server_name": self.settings.host,
            "server_port": self.settings.port,
            "share": self.settings.enable_sharing,
            "favicon_path": str(favicon) if favicon else None,
            "allowed_paths": allowed,
        }
        if self.settings.require_auth:
            launch_kwargs["auth"] = self.authenticate
        demo.launch(**launch_kwargs)
