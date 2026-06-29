"""End-to-end clothing inpainting (segmentation, pose, diffusion, blend)."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from outfit_studio.config import get_settings
from outfit_studio.constants import (
    INPAINT_STRENGTH,
    LATENT_ALIGN,
    MASK_ON,
    MIN_INSTANCE_CLOTHES_PIXELS,
    MIN_POSE_IMAGE_SIDE,
    SEED_MAX,
    GenerateProgress,
    PersonProgress,
)
from outfit_studio.content_config import get_default_negative_prompt, get_default_prompt
from outfit_studio.ml.gpu_memory import free_cuda_cache, release_segmentation_gpu
from outfit_studio.ml.inpainter import get_inpaint_engine
from outfit_studio.ml.pipeline_debug import PipelineDebugSession
from outfit_studio.ml.pose import get_pose_estimator
from outfit_studio.ml.segmentation import run_segmentation
from outfit_studio.utils.image import (
    align_masks,
    apply_reflection_padding,
    blend_images_with_enhancements,
    composite_crop_onto,
    get_bounding_box,
    get_crop_info,
    prepare_instance_masks,
    remove_reflection_padding,
)
from outfit_studio.utils.logging import log_duration

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(_fraction: float, _desc: str) -> None:
    pass


def _scoped_progress(
    report: ProgressCallback,
    span: tuple[float, float],
    prefix: str,
) -> ProgressCallback:
    """Map local 0–1 progress into a sub-range of the overall bar."""
    start, end = span

    def sub(local: float, desc: str) -> None:
        report(start + local * (end - start), f"{prefix}: {desc}")

    return sub


class GenerationPipeline:
    """Coordinates segmentation → per-person inpaint → blend (original flow)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _process_single_mask(
        self,
        full_image: Image.Image,
        person_mask: np.ndarray,
        clothes_mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        generator: torch.Generator,
        model: str | None,
        use_controlnet: bool,
        progress: ProgressCallback | None = None,
        progress_span: tuple[float, float] = (0.0, 1.0),
        person_index: int = 1,
        person_total: int = 1,
        debug: PipelineDebugSession | None = None,
    ) -> tuple[Image.Image, dict]:
        report = progress or _noop_progress
        prefix = f"Person {person_index}/{person_total}"
        sub = _scoped_progress(report, progress_span, prefix)

        sub(PersonProgress.PREP, "Preparing region")
        clothes_alpha = Image.fromarray((clothes_mask * MASK_ON).astype(np.uint8))
        person_alpha = Image.fromarray((person_mask * MASK_ON).astype(np.uint8))

        person_binary = person_alpha.point(lambda p: MASK_ON if p > 0 else 0)
        clothes_binary = clothes_alpha.point(lambda p: MASK_ON if p > 0 else 0)
        combined_mask = Image.new("L", full_image.size, 0)
        combined_mask.paste(person_binary, (0, 0))
        combined_mask.paste(clothes_binary, (0, 0), clothes_binary)

        crop_info = get_crop_info(combined_mask)
        logger.debug(
            "Instance crop box (%d,%d)-(%d,%d) infer_size target from mask",
            crop_info["left"],
            crop_info["top"],
            crop_info["right"],
            crop_info["bottom"],
        )
        crop_box = (
            crop_info["left"],
            crop_info["top"],
            crop_info["right"],
            crop_info["bottom"],
        )
        cropped_image = full_image.crop(crop_box)
        cropped_clothes = clothes_alpha.crop(crop_box)
        cropped_person_mask = person_alpha.crop(crop_box)

        target_size = max(cropped_image.size)
        if cropped_image.size[0] != cropped_image.size[1]:
            padded_image, padding_info = apply_reflection_padding(
                cropped_image, (target_size, target_size), center=crop_info["center"]
            )
            padded_mask, _ = apply_reflection_padding(
                cropped_clothes, (target_size, target_size), center=crop_info["center"]
            )
        else:
            padded_image = cropped_image
            padded_mask = cropped_clothes
            padding_info = None

        cnet_image = padded_image.copy()
        binary_mask = padded_mask.point(lambda p: MASK_ON if p > 0 else 0)
        cnet_image.paste(0, (0, 0), binary_mask)
        cnet_image = cnet_image.convert("RGB")

        pose_est = get_pose_estimator()
        control_image = None
        if use_controlnet:
            sub(PersonProgress.POSE_DETECT, "Detecting pose")
            if cnet_image.width >= MIN_POSE_IMAGE_SIDE and cnet_image.height >= MIN_POSE_IMAGE_SIDE:
                bboxes = pose_est.get_bboxes(cnet_image)
                logger.debug("Pose bboxes for crop: %d", len(bboxes))
                sub(PersonProgress.POSE_GUIDE, "Building ControlNet guide")
                control_image = pose_est.estimate(cnet_image, bboxes=bboxes)
            else:
                logger.warning(
                    "Skipping ControlNet pose for degenerate crop %dx%d",
                    cnet_image.width,
                    cnet_image.height,
                )
                sub(PersonProgress.PREP_AREA, "Preparing inpaint area (no pose)")
        else:
            sub(PersonProgress.PREP_AREA, "Preparing inpaint area")

        top, left, bottom, right = get_bounding_box(np.array(binary_mask) > 0)
        engine = get_inpaint_engine()
        sub(PersonProgress.LOAD_MODEL, "Loading model")
        engine.load(model, use_controlnet)
        infer_size = max(
            min(max(right - left, bottom - top), self.settings.inference_size)
            // LATENT_ALIGN
            * LATENT_ALIGN,
            self.settings.min_inference_size,
        )
        resolved_model = model or get_inpaint_engine().default_model_id()
        logger.info(
            "Inpainting crop %dx%d at %dx%d (model=%s, controlnet=%s)",
            cnet_image.width,
            cnet_image.height,
            infer_size,
            infer_size,
            resolved_model,
            use_controlnet,
        )

        if debug is not None:
            prefix = f"person_{person_index:02d}"
            debug.save_mask(f"{prefix}/person_mask.png", person_mask)
            debug.save_mask(f"{prefix}/clothes_mask.png", clothes_mask)
            debug.save_image(f"{prefix}/01_crop_source.png", cropped_image)
            debug.save_image(f"{prefix}/02_padded_image.png", padded_image)
            debug.save_mask(f"{prefix}/02_padded_clothes_mask.png", np.array(binary_mask) > 0)
            debug.save_image(f"{prefix}/03_inpaint_input.png", cnet_image)
            if control_image is not None:
                debug.save_image(f"{prefix}/04_controlnet_pose.png", control_image)
            debug.record(
                f"person_{person_index}",
                model=resolved_model,
                use_controlnet=use_controlnet,
                infer_size=infer_size,
                crop_box=crop_box,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                inference_steps=num_inference_steps,
                strength=INPAINT_STRENGTH,
            )

        def on_diffusion_step(step: int, total: int) -> None:
            local = PersonProgress.DIFFUSION_START + PersonProgress.DIFFUSION_SPAN * (
                step + 1
            ) / max(total, 1)
            sub(local, f"Diffusion step {step + 1}/{total}")

        sub(PersonProgress.DIFFUSION_START, f"Starting diffusion ({num_inference_steps} steps)")
        with log_duration(logger, "diffusion inpaint", steps=num_inference_steps):
            output_image = engine.inpaint(
                cnet_image,
                binary_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                control_image=control_image,
                width=infer_size,
                height=infer_size,
                strength=INPAINT_STRENGTH,
                on_step=on_diffusion_step,
            )
        if output_image.size != cnet_image.size:
            output_image = output_image.resize(cnet_image.size, Image.LANCZOS)

        if debug is not None:
            prefix = f"person_{person_index:02d}"
            debug.save_image(f"{prefix}/05_diffusion_output.png", output_image)

        sub(PersonProgress.BLEND, "Blending result")
        reflection_stripped = remove_reflection_padding(output_image, padding_info)
        logger.debug("Reflection padding removed")
        result_image = blend_images_with_enhancements(
            cropped_image,
            reflection_stripped,
            cropped_clothes,
            cropped_person_mask,
        )
        logger.debug("Feathered blend completed for instance")
        if debug is not None:
            prefix = f"person_{person_index:02d}"
            debug.save_image(f"{prefix}/06_blended_crop.png", result_image)
        sub(1.0, "Region complete")
        return result_image, crop_info

    def generate(
        self,
        image: Image.Image,
        person_mask: np.ndarray | None = None,
        clothes_mask: np.ndarray | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        model: str | None = None,
        use_controlnet: bool | None = None,
        username: str = "guest",
        progress: ProgressCallback | None = None,
        debug_session_dir: str | None = None,
    ) -> tuple[Image.Image, str]:
        report = progress or _noop_progress
        report(GenerateProgress.PREP_START, "Preparing image and masks")
        logger.info(
            "Generate started (user=%s, model=%s, steps=%s, cfg=%s, controlnet=%s)",
            username,
            model or "default",
            steps,
            guidance_scale,
            use_controlnet,
        )
        release_segmentation_gpu()
        free_cuda_cache()

        image = image.convert("RGB")
        w, h = image.size
        logger.debug("Source image %dx%d", w, h)

        if person_mask is None or clothes_mask is None:
            logger.info("No editor masks — running full segmentation")
            report(GenerateProgress.SEGMENT, "Running clothes segmentation")
            person_mask, clothes_mask, _ = run_segmentation(
                image,
                settings=self.settings,
                username=username,
                debug_session_dir=debug_session_dir,
            )

        person_mask, clothes_mask = align_masks(person_mask, clothes_mask, h, w)

        prompt = prompt or get_default_prompt()
        negative_prompt = negative_prompt or get_default_negative_prompt()
        steps = steps or self.settings.inpaint_steps
        guidance_scale = guidance_scale or self.settings.guidance_scale
        use_controlnet = (
            use_controlnet if use_controlnet is not None else self.settings.use_controlnet
        )
        seed = seed if seed is not None else random.randint(0, SEED_MAX)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using seed %d on %s", seed, device)
        generator = torch.Generator(device=device).manual_seed(seed)

        session, active_dir = PipelineDebugSession.open_or_create(
            self.settings, username, debug_session_dir
        )
        debug = session.subfolder("generation") if session else None
        if debug is not None:
            debug.save_image("00_source.png", image)
            debug.metadata.update(
                {
                    "username": username,
                    "seed": seed,
                    "device": device,
                    "model": model,
                    "inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "use_controlnet": use_controlnet,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )

        pose_est = get_pose_estimator()
        report(GenerateProgress.DETECT_PEOPLE, "Detecting people")
        bboxes = pose_est.get_bboxes(image)
        instances = prepare_instance_masks(person_mask, clothes_mask, bboxes)
        if debug is not None:
            debug.save_mask("01_person_mask.png", person_mask)
            debug.save_mask("02_clothes_mask.png", clothes_mask)
            debug.save_overlay("03_masks_overlay.png", image, person_mask, clothes_mask)
            debug.metadata["bboxes"] = bboxes.tolist() if len(bboxes) else []
            debug.metadata["instance_count"] = len(instances)
        if not instances:
            logger.warning("No per-person instances from bboxes — using full-frame masks")
            instances = [(person_mask, clothes_mask)]

        active = [inst for inst in instances if int(inst[1].sum()) >= MIN_INSTANCE_CLOTHES_PIXELS]
        if not active:
            active = instances

        report(GenerateProgress.PREP_END, "Loading inpainting model")
        get_inpaint_engine().load(model, use_controlnet)
        if debug is not None:
            engine = get_inpaint_engine()
            debug.metadata["loaded_model"] = engine._current_model
            debug.metadata["model_architecture"] = engine._architecture
            debug.metadata["controlnet_active"] = engine._use_controlnet

        full_image = image.copy()
        logger.info("Processing %d person instance(s)", len(active))
        for idx, (person_m, clothes_m) in enumerate(active, start=1):
            clothes_px = int(clothes_m.sum())
            logger.info(
                "Instance %d/%d — person_px=%d clothes_px=%d",
                idx,
                len(active),
                int(person_m.sum()),
                clothes_px,
            )
            if clothes_px == 0:
                logger.info("Instance %d/%d skipped — no clothes mask in bbox", idx, len(active))
                continue
            span_start = GenerateProgress.PERSON_START + GenerateProgress.PERSON_SPAN * (
                idx - 1
            ) / max(len(active), 1)
            span_end = GenerateProgress.PERSON_START + GenerateProgress.PERSON_SPAN * idx / max(
                len(active), 1
            )
            result_image, crop_info = self._process_single_mask(
                full_image,
                person_m,
                clothes_m,
                prompt,
                negative_prompt,
                guidance_scale,
                steps,
                generator,
                model,
                use_controlnet,
                progress=report,
                progress_span=(span_start, span_end),
                person_index=idx,
                person_total=len(active),
                debug=debug,
            )
            full_image = composite_crop_onto(
                full_image,
                result_image,
                crop_info["left"],
                crop_info["top"],
            )
            if debug is not None:
                debug.save_image(f"person_{idx:02d}/07_composited_full.png", full_image)
            logger.info("Pasted inpainted region at (%d, %d)", crop_info["left"], crop_info["top"])

        pose_est.unload()
        free_cuda_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        report(GenerateProgress.SAVE, "Saving result")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_{ts}.png"
        out_path = self.settings.resolved_output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_image.save(out_path)
        logger.info("Saved output → %s", out_path)
        if debug is not None:
            debug.save_image("99_final_output.png", full_image)
            debug.metadata["output_file"] = filename
            debug.save_meta()
            logger.info("Pipeline debug artifacts → %s", active_dir or debug.root)
        report(1.0, "Complete")
        return full_image, filename, active_dir


@lru_cache
def get_pipeline() -> GenerationPipeline:
    return GenerationPipeline()
