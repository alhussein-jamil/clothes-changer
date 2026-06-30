#!/usr/bin/env python3
"""Benchmark inpaint step counts vs a high-step reference."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OUTFIT_STUDIO_TORCH_COMPILE", "false")
os.environ.setdefault("OUTFIT_STUDIO_REQUIRE_AUTH", "false")

from outfit_studio.config import get_settings  # noqa: E402
from outfit_studio.content_config import (  # noqa: E402
    get_default_negative_prompt,
    get_default_prompt,
)
from outfit_studio.ml.inpainter import InpaintEngine  # noqa: E402
from outfit_studio.ml.pose import PoseEstimator  # noqa: E402
from outfit_studio.ml.segmentation_workflow import run_segmentation  # noqa: E402

REFERENCE_STEPS = 60
STEP_COUNTS = (8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56)
SEED = 4242
SSIM_PLATEAU_DELTA = 0.003  # marginal gain below this → diminishing returns
SSIM_TARGET_RATIO = 0.985  # within 1.5% of best achievable SSIM


@dataclass
class StepResult:
    steps: int
    seconds: float
    ssim_masked: float
    mae_masked: float
    ssim_gain_vs_prev: float | None


def _masked_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """SSIM and MAE inside the binary mask (RGB, 0–255)."""
    m = mask > 0
    if not m.any():
        return 1.0, 0.0

    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    mae = float(np.abs(ref[m] - cand[m]).mean())

    # SSIM on a tight crop around the mask for stability.
    ys, xs = np.where(m)
    pad = 8
    top = max(0, int(ys.min()) - pad)
    bottom = min(reference.shape[0], int(ys.max()) + pad + 1)
    left = max(0, int(xs.min()) - pad)
    right = min(reference.shape[1], int(xs.max()) + pad + 1)
    ref_crop = ref[top:bottom, left:right]
    cand_crop = cand[top:bottom, left:right]
    win = min(7, min(ref_crop.shape[0], ref_crop.shape[1]))
    if win < 3:
        return 1.0, mae
    score = float(
        ssim(
            ref_crop,
            cand_crop,
            channel_axis=2,
            data_range=255.0,
            win_size=win if win % 2 == 1 else win - 1,
        )
    )
    return score, mae


def _prepare_inpaint(
    image: Image.Image,
    person: np.ndarray,
    clothes: np.ndarray,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    """Build inpaint input matching the pipeline crop (no full generate)."""
    from outfit_studio.ml.pipeline import GenerationPipeline

    pipeline = GenerationPipeline()
    person_binary = Image.fromarray((person > 0).astype(np.uint8) * 255, mode="L")
    clothes_binary = Image.fromarray((clothes > 0).astype(np.uint8) * 255, mode="L")
    combined = Image.new("L", image.size, 0)
    combined.paste(person_binary, (0, 0))
    combined.paste(clothes_binary, (0, 0), clothes_binary)

    from outfit_studio.utils.image import get_crop_info

    crop_info = get_crop_info(combined)
    crop_box = (
        crop_info["left"],
        crop_info["top"],
        crop_info["right"],
        crop_info["bottom"],
    )
    cropped = image.crop(crop_box)
    cropped_clothes = clothes_binary.crop(crop_box)

    target_size = max(cropped.size)
    if cropped.size[0] != cropped.size[1]:
        from outfit_studio.utils.image import apply_reflection_padding

        padded_image, _ = apply_reflection_padding(
            cropped, (target_size, target_size), center=crop_info["center"]
        )
        padded_mask, _ = apply_reflection_padding(
            cropped_clothes, (target_size, target_size), center=crop_info["center"]
        )
    else:
        padded_image = cropped
        padded_mask = cropped_clothes

    binary_mask = Image.fromarray(((np.array(padded_mask) > 0).astype(np.uint8) * 255), mode="L")
    inpaint_input = padded_image.copy()
    inpaint_input.paste(0, (0, 0), binary_mask)

    control_image = None
    if pipeline.settings.content.use_controlnet:
        pose = PoseEstimator()
        kp, sc = pose.estimate_keypoints(inpaint_input.convert("RGB"))
        control_image = pose.render_skeleton(inpaint_input.size, kp, sc)
        pose.unload()

    return inpaint_input.convert("RGB"), binary_mask, control_image


def _recommend(results: list[StepResult]) -> int:
    if not results:
        return REFERENCE_STEPS
    best_ssim = max(r.ssim_masked for r in results)
    target = best_ssim * SSIM_TARGET_RATIO
    plateau: int | None = None
    prev = results[0]
    for row in results[1:]:
        gain = row.ssim_masked - prev.ssim_masked
        if gain < SSIM_PLATEAU_DELTA and row.ssim_masked >= target:
            plateau = prev.steps
            break
        prev = row
    if plateau is not None:
        return plateau
    for row in results:
        if row.ssim_masked >= target:
            return row.steps
    return results[-1].steps


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    out_dir = settings.resolved_output_dir / "steps_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = PROJECT_ROOT / "docs/assets/demo-source.png"
    image = Image.open(source_path).convert("RGB")
    print(f"Source: {source_path.name} ({image.size[0]}x{image.size[1]})")

    print("Running segmentation for masks …")
    person, clothes, _ = run_segmentation(image, settings=settings, username="benchmark")
    inpaint_input, mask, control = _prepare_inpaint(image, person, clothes)
    mask_np = (np.array(mask) > 0).astype(np.uint8)
    print(f"Inpaint crop: {inpaint_input.size[0]}x{inpaint_input.size[1]}")
    print(
        f"Model={settings.content.default_inpaint} controlnet={settings.content.use_controlnet} "
        f"cfg={settings.content.guidance_scale}"
    )

    engine = InpaintEngine(settings)
    engine.load(settings.content.default_inpaint, settings.content.use_controlnet)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = get_default_prompt()
    negative = get_default_negative_prompt()

    def run_steps(steps: int) -> tuple[Image.Image, float]:
        gen = torch.Generator(device=device).manual_seed(SEED)
        t0 = time.perf_counter()
        out = engine.inpaint(
            inpaint_input,
            mask,
            prompt=prompt,
            negative_prompt=negative,
            steps=steps,
            guidance_scale=settings.content.guidance_scale,
            generator=gen,
            control_image=control,
        )
        return out, time.perf_counter() - t0

    print(f"\nGenerating reference @ {REFERENCE_STEPS} steps …")
    ref_img, ref_time = run_steps(REFERENCE_STEPS)
    ref_np = np.array(ref_img.convert("RGB"))
    ref_img.save(out_dir / f"reference_{REFERENCE_STEPS}.png")

    results: list[StepResult] = []
    prev_ssim: float | None = None
    print(f"\n{'steps':>5}  {'sec':>6}  {'SSIM':>7}  {'MAE':>7}  {'dSSIM':>7}")
    print("-" * 40)
    for steps in STEP_COUNTS:
        img, elapsed = run_steps(steps)
        arr = np.array(img.convert("RGB"))
        ssim_score, mae = _masked_metrics(ref_np, arr, mask_np)
        gain = None if prev_ssim is None else ssim_score - prev_ssim
        prev_ssim = ssim_score
        row = StepResult(steps, elapsed, ssim_score, mae, gain)
        results.append(row)
        img.save(out_dir / f"steps_{steps:02d}.png")
        gain_txt = f"{gain:+.4f}" if gain is not None else "   —"
        print(f"{steps:5d}  {elapsed:6.2f}  {ssim_score:7.4f}  {mae:7.2f}  {gain_txt:>7}")

    recommended = _recommend(results)
    ref_row = next(r for r in results if r.steps == 40)
    summary = {
        "reference_steps": REFERENCE_STEPS,
        "reference_seconds": ref_time,
        "model": settings.content.default_inpaint,
        "use_controlnet": settings.content.use_controlnet,
        "guidance_scale": settings.content.guidance_scale,
        "current_config_steps": settings.content.steps,
        "recommended_steps": recommended,
        "metrics_at_40": asdict(ref_row),
        "all_results": [asdict(r) for r in results],
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nReference runtime ({REFERENCE_STEPS} steps): {ref_time:.1f}s")
    print(f"Current config steps: {settings.content.steps}")
    print(f"Recommended steps (SSIM plateau): {recommended}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
