"""Generation event handlers for GradioApp."""

from __future__ import annotations

import logging
import random

import gradio as gr
import numpy as np
from PIL import Image

from outfit_studio.constants import SEED_MAX, GenerateProgress
from outfit_studio.content_config import get_default_negative_prompt, get_default_prompt
from outfit_studio.ml.inpainter import get_inpaint_engine
from outfit_studio.ml.segmentation_workflow import run_segmentation
from outfit_studio.ui.masks import masks_have_pixels, resolve_masks_for_generate
from outfit_studio.ui.operation_control import OperationCancelled, bind_request

logger = logging.getLogger(__name__)


class GenerationHandlersMixin:
    def _compose_generation_params(
        self,
        *,
        is_admin: bool,
        prompt: str,
        negative_prompt: str,
        user_prompt_addon: str,
        model_id: str,
        use_controlnet: bool,
        steps: int,
        guidance_scale: float,
        seed: int,
        random_seed: bool,
    ) -> dict[str, object]:
        content = self.settings.content
        if is_admin:
            full_prompt = (prompt or "").strip()
            if not full_prompt:
                raise gr.Error("Prompt cannot be empty")
            return {
                "prompt": full_prompt,
                "negative_prompt": (negative_prompt or "").strip(),
                "model_id": model_id if model_id in self.model_ids else self.default_model,
                "use_controlnet": use_controlnet,
                "steps": int(steps),
                "guidance_scale": float(guidance_scale),
                "seed": random.randint(0, SEED_MAX) if random_seed else int(seed),
            }

        base = get_default_prompt().strip()
        addon = (user_prompt_addon or "").strip()
        full_prompt = f"{addon}, {base}" if addon else base
        return {
            "prompt": full_prompt,
            "negative_prompt": get_default_negative_prompt().strip(),
            "model_id": self.default_model,
            "use_controlnet": content.use_controlnet,
            "steps": content.steps,
            "guidance_scale": content.guidance_scale,
            "seed": random.randint(0, SEED_MAX),
        }

    def generate(
        self,
        editor: dict | None,
        clean_source: Image.Image | None,
        segment_key: str | None,
        segment_masks: tuple[np.ndarray, np.ndarray] | None,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        use_controlnet: bool,
        steps: int,
        guidance_scale: float,
        seed: int,
        random_seed: bool,
        debug_session_dir: str | None,
        user_prompt_addon: str,
        request: gr.Request,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[tuple[Image.Image, Image.Image] | None, int, str | None]:
        bind_request(request)
        username = self._session_username(request)
        user = self.db.get_user(username) if username else None
        is_admin = bool(user and user.is_admin)
        params = self._compose_generation_params(
            is_admin=is_admin,
            prompt=prompt,
            negative_prompt=negative_prompt,
            user_prompt_addon=user_prompt_addon,
            model_id=model_id,
            use_controlnet=use_controlnet,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            random_seed=random_seed,
        )
        resolved_prompt = str(params["prompt"])
        resolved_negative = str(params["negative_prompt"])
        model_id = str(params["model_id"])
        use_controlnet = bool(params["use_controlnet"])
        steps = int(params["steps"])
        guidance_scale = float(params["guidance_scale"])
        actual_seed = int(params["seed"])

        if not username:
            raise gr.Error("Not authenticated")
        if not user:
            raise gr.Error("User not found")
        if not is_admin and user.credits <= 0:
            raise gr.Error("No credits remaining. Contact an administrator.")

        if not is_admin:
            debug_session_dir = None

        engine = get_inpaint_engine()
        engine.wait_for_preload(progress=lambda fraction, desc: progress(fraction, desc=desc))

        progress(0, desc="Preparing generation")

        pipeline_image = self._pipeline_source(editor, clean_source, segment_key)
        if pipeline_image is None:
            return None, seed, debug_session_dir
        source = pipeline_image

        person_mask, clothes_mask = resolve_masks_for_generate(editor, segment_masks, source)

        try:
            if not masks_have_pixels(person_mask, clothes_mask):
                progress(GenerateProgress.PREP_START, desc="Running clothes segmentation")
                person_mask, clothes_mask, active_dir = run_segmentation(
                    source,
                    settings=self.settings,
                    username=username,
                    debug_session_dir=debug_session_dir,
                )
                debug_session_dir = active_dir

            def report_progress(fraction: float, desc: str) -> None:
                progress(fraction, desc=desc)

            result, filename, active_debug_dir = self.pipeline.generate(
                image=source,
                person_mask=person_mask,
                clothes_mask=clothes_mask,
                prompt=resolved_prompt,
                negative_prompt=resolved_negative,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=actual_seed,
                model=model_id,
                use_controlnet=use_controlnet,
                username=username,
                progress=report_progress,
                debug_session_dir=debug_session_dir,
            )
        except OperationCancelled:
            return gr.update(), seed, debug_session_dir
        except Exception as e:
            logger.exception("Generation failed")
            message = str(e).strip() or type(e).__name__
            raise gr.Error(message) from e

        if not is_admin:
            self.db.deduct_credit(username)

        if is_admin:
            log_prompt = f"+: {resolved_prompt} | -: {resolved_negative}"
        else:
            addon = (user_prompt_addon or "").strip()
            log_prompt = addon if addon else "(default)"
        self.db.log_image(username, filename, log_prompt)

        debug_dir = active_debug_dir if is_admin else None
        return gr.update(value=(source, result.convert("RGB"))), actual_seed, debug_dir
