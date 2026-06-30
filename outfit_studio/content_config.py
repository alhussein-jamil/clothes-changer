"""Branded copy, prompts, and ML defaults from YAML (local overrides default)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DEFAULT_FILE = _CONFIG_DIR / "content.default.yaml"
_LOCAL_FILE = _CONFIG_DIR / "content.local.yaml"


class ContentSettings(BaseModel):
    """Typed view of config/content*.yaml — single source for ML and branding defaults."""

    app_name: str = "Outfit Studio"
    tagline: str = ""
    default_prompt: str = ""
    negative_prompt: str = ""
    default_inpaint: str = "runwayml/stable-diffusion-inpainting"
    human_parser: str = "fashn-ai/fashn-human-parser"
    controlnet: str = "lllyasviel/sd-controlnet-openpose"
    use_controlnet: bool = True
    steps: int = 50
    guidance_scale: float = 6.5
    inference_size: int = 512
    detection_threshold: float = 0.5
    keypoint_threshold: float = 0.3
    pose_mode: str = "balanced"
    clothes_confidence: float = 0.15
    min_component_area: int = 32
    clothes_edge_grow_px: int = 7
    checkpoint_urls: dict[str, str] = Field(default_factory=dict)
    model_aliases: dict[str, list[str]] = Field(default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config() -> dict[str, Any]:
    data: dict[str, Any] = {}
    sources: list[str] = []
    if _DEFAULT_FILE.is_file():
        data = yaml.safe_load(_DEFAULT_FILE.read_text(encoding="utf-8")) or {}
        sources.append("default")
    if _LOCAL_FILE.is_file():
        local = yaml.safe_load(_LOCAL_FILE.read_text(encoding="utf-8")) or {}
        data = _deep_merge(data, local)
        sources.append("local")
    logger.debug(
        "Content config loaded from %s (%d top-level keys)",
        "+".join(sources) if sources else "built-in defaults",
        len(data),
    )
    return data


def _parse_content_settings(data: dict[str, Any]) -> ContentSettings:
    app = data.get("app", {}) if isinstance(data.get("app"), dict) else {}
    prompts = data.get("prompts", {}) if isinstance(data.get("prompts"), dict) else {}
    models = data.get("models", {}) if isinstance(data.get("models"), dict) else {}
    generation = data.get("generation", {}) if isinstance(data.get("generation"), dict) else {}
    pose = data.get("pose", {}) if isinstance(data.get("pose"), dict) else {}
    segmentation_raw = data.get("segmentation")
    segmentation = segmentation_raw if isinstance(segmentation_raw, dict) else {}

    human_parser = models.get("human_parser")
    if human_parser is None:
        human_parser = "fashn-ai/fashn-human-parser"

    urls = models.get("download_urls", {})
    aliases = models.get("aliases", {})

    return ContentSettings(
        app_name=str(app.get("name", "Outfit Studio")),
        tagline=str(app.get("tagline", "")),
        default_prompt=str(prompts.get("default", "")),
        negative_prompt=str(prompts.get("negative", "")),
        default_inpaint=str(models.get("default_inpaint", "runwayml/stable-diffusion-inpainting")),
        human_parser=str(human_parser),
        controlnet=str(models.get("controlnet", "lllyasviel/sd-controlnet-openpose")),
        use_controlnet=bool(generation.get("use_controlnet", True)),
        steps=int(generation.get("steps", 50)),
        guidance_scale=float(generation.get("guidance_scale", 6.5)),
        inference_size=int(generation.get("inference_size", 512)),
        detection_threshold=float(pose.get("detection_threshold", 0.5)),
        keypoint_threshold=float(pose.get("keypoint_threshold", 0.3)),
        pose_mode=str(pose.get("mode", "balanced")),
        clothes_confidence=float(segmentation.get("clothes_confidence", 0.15)),
        min_component_area=int(segmentation.get("min_component_area", 32)),
        clothes_edge_grow_px=int(segmentation.get("clothes_edge_grow_px", 7)),
        checkpoint_urls={str(k): str(v) for k, v in urls.items()} if isinstance(urls, dict) else {},
        model_aliases={
            str(name): [str(alias) for alias in alias_list] for name, alias_list in aliases.items()
        }
        if isinstance(aliases, dict)
        else {},
    )


@lru_cache
def get_content_settings() -> ContentSettings:
    return _parse_content_settings(_load_raw_config())


def clear_content_config_cache() -> None:
    get_content_settings.cache_clear()


# Thin accessors — keep call sites readable without reaching into nested fields everywhere.


def get_app_name() -> str:
    return get_content_settings().app_name


def get_tagline() -> str:
    return get_content_settings().tagline


def get_default_prompt() -> str:
    return get_content_settings().default_prompt


def get_default_negative_prompt() -> str:
    return get_content_settings().negative_prompt


def get_default_inpaint_model() -> str:
    return get_content_settings().default_inpaint


def get_human_parser_model() -> str:
    return get_content_settings().human_parser


def get_controlnet_model() -> str:
    return get_content_settings().controlnet


def get_use_controlnet() -> bool:
    return get_content_settings().use_controlnet


def get_inpaint_steps() -> int:
    return get_content_settings().steps


def get_guidance_scale() -> float:
    return get_content_settings().guidance_scale


def get_inference_size() -> int:
    return get_content_settings().inference_size


def get_detection_threshold() -> float:
    return get_content_settings().detection_threshold


def get_pose_keypoint_threshold() -> float:
    return get_content_settings().keypoint_threshold


def get_pose_mode() -> str:
    return get_content_settings().pose_mode


def get_segmentation_clothes_confidence() -> float:
    return get_content_settings().clothes_confidence


def get_segmentation_min_component_area() -> int:
    return get_content_settings().min_component_area


def get_segmentation_clothes_edge_grow_px() -> int:
    return get_content_settings().clothes_edge_grow_px


def get_checkpoint_urls() -> dict[str, str]:
    return dict(get_content_settings().checkpoint_urls)


def get_model_aliases() -> dict[str, list[str]]:
    return dict(get_content_settings().model_aliases)
