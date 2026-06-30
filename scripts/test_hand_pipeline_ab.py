#!/usr/bin/env python3
"""End-to-end pipeline A/B: clothing inpaint with vs without hand protect."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OUTFIT_STUDIO_TORCH_COMPILE", "false")
os.environ.setdefault("OUTFIT_STUDIO_REQUIRE_AUTH", "false")
os.environ.setdefault("OUTFIT_STUDIO_PIPELINE_DEBUG", "true")

from outfit_studio.config import get_settings
from outfit_studio.content_config import clear_content_config_cache
from outfit_studio.ml.pipeline import GenerationPipeline
from outfit_studio.ml.pose import PoseEstimator
from outfit_studio.utils.hand_mask import build_combined_hand_mask


def _write_hands_config(tmp_dir: Path, *, protect: bool) -> None:
    cfg_dir = tmp_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    default = yaml.safe_load((PROJECT_ROOT / "config/content.default.yaml").read_text())
    default["hands"] = {
        **default.get("hands", {}),
        "protect": protect,
    }
    default["generation"] = {
        **default.get("generation", {}),
        "steps": 24,
    }
    default["models"] = {
        **default.get("models", {}),
        "default_inpaint": "runwayml/stable-diffusion-inpainting",
    }
    (cfg_dir / "content.default.yaml").write_text(yaml.dump(default))
    (cfg_dir / "content.local.yaml").write_text("")


def _expanded_masks(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    w, h = image.size
    person = np.zeros((h, w), dtype=np.uint8)
    clothes = np.zeros((h, w), dtype=np.uint8)
    person[h // 8 : h - h // 16, w // 4 : 3 * w // 4] = 1
    # Shirt region overlapping both hands — the failure mode we target.
    clothes[h // 6 : int(h * 0.78), w // 5 : 4 * w // 5] = 1
    return person, clothes


def _overlap_px(person: np.ndarray, clothes: np.ndarray, image: Image.Image) -> int:
    pose = PoseEstimator()
    kp, sc = pose.estimate_keypoints(image)
    hand_mask = build_combined_hand_mask(kp, sc, (image.height, image.width), kpt_thr=0.3)
    pose.unload()
    return int((clothes.astype(bool) & (hand_mask > 0)).sum())


def _run(
    label: str,
    tmp_dir: Path,
    image: Image.Image,
    person: np.ndarray,
    clothes: np.ndarray,
) -> Path:
    import outfit_studio.content_config as cc

    monkey_cfg = tmp_dir / label
    _write_hands_config(monkey_cfg, protect="on" in label)
    cc._CONFIG_DIR = monkey_cfg / "config"
    cc._DEFAULT_FILE = cc._CONFIG_DIR / "content.default.yaml"
    cc._LOCAL_FILE = cc._CONFIG_DIR / "content.local.yaml"
    clear_content_config_cache()
    get_settings.cache_clear()
    from outfit_studio.ml.inpainter import get_inpaint_engine
    from outfit_studio.ml.pipeline import get_pipeline

    get_pipeline.cache_clear()
    get_inpaint_engine.cache_clear()

    out_debug = get_settings().resolved_output_dir / f"pipeline_ab_{label}"
    out_debug.mkdir(parents=True, exist_ok=True)

    pipeline = GenerationPipeline()
    result, filename, debug_dir = pipeline.generate(
        image,
        person_mask=person,
        clothes_mask=clothes,
        prompt="high quality photo, white cotton t-shirt, natural lighting, realistic fabric",
        negative_prompt=(
            "extra fingers, missing fingers, fused fingers, malformed hands, bad anatomy"
        ),
        seed=4242,
        model="runwayml/stable-diffusion-inpainting",
        use_controlnet=True,
        username="hand_ab",
        debug_session_dir=str(out_debug),
    )
    out_path = out_debug / f"result_{label}.png"
    result.save(out_path)
    print(f"{label}: saved {out_path} (debug={debug_dir})")
    return out_path


def main() -> None:
    get_settings.cache_clear()
    clear_content_config_cache()
    settings = get_settings()
    settings.ensure_dirs()

    source = Image.open(PROJECT_ROOT / "docs/assets/demo-source.png").convert("RGB")
    person, clothes = _expanded_masks(source)
    overlap = _overlap_px(person, clothes, source)
    print(f"Clothes/hand mask overlap: {overlap} px")

    tmp = settings.resolved_output_dir / "hand_ab_configs"
    tmp.mkdir(parents=True, exist_ok=True)

    without = _run("hands_off", tmp, source, person, clothes)
    with_fix = _run("hands_on", tmp, source, person, clothes)
    print(f"Compare:\n  without: {without}\n  with:    {with_fix}")


if __name__ == "__main__":
    main()
