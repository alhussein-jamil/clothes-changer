"""Branded copy, prompts, and ML defaults from YAML (local overrides default)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DEFAULT_FILE = _CONFIG_DIR / "content.default.yaml"
_LOCAL_FILE = _CONFIG_DIR / "content.local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache
def get_content_config() -> dict[str, Any]:
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


def clear_content_config_cache() -> None:
    get_content_config.cache_clear()


def _section(name: str) -> dict[str, Any]:
    value = get_content_config().get(name, {})
    return value if isinstance(value, dict) else {}


def _models() -> dict[str, Any]:
    return _section("models")


def _generation() -> dict[str, Any]:
    return _section("generation")


def _pose() -> dict[str, Any]:
    return _section("pose")


def get_app_name() -> str:
    return str(get_content_config().get("app", {}).get("name", "Outfit Studio"))


def get_tagline() -> str:
    return str(get_content_config().get("app", {}).get("tagline", ""))


def get_default_prompt() -> str:
    return str(get_content_config().get("prompts", {}).get("default", ""))


def get_default_negative_prompt() -> str:
    return str(get_content_config().get("prompts", {}).get("negative", ""))


def get_default_inpaint_model() -> str:
    return str(_models().get("default_inpaint", "runwayml/stable-diffusion-inpainting"))


def get_segformer_model() -> str:
    return str(_models().get("segformer", "mattmdjaga/segformer_b2_clothes"))


def get_extra_clothes_model() -> str:
    return str(_models().get("extra_clothes", "cloth_segm.pth"))


def get_controlnet_model() -> str:
    return str(_models().get("controlnet", "lllyasviel/sd-controlnet-openpose"))


def get_use_controlnet() -> bool:
    return bool(_generation().get("use_controlnet", True))


def get_inpaint_steps() -> int:
    return int(_generation().get("steps", 50))


def get_guidance_scale() -> float:
    return float(_generation().get("guidance_scale", 6.5))


def get_inference_size() -> int:
    return int(_generation().get("inference_size", 512))


def get_min_inference_size() -> int:
    return int(_generation().get("min_inference_size", 256))


def get_detection_threshold() -> float:
    return float(_pose().get("detection_threshold", 0.5))


def get_pose_keypoint_threshold() -> float:
    return float(_pose().get("keypoint_threshold", 0.3))


def get_pose_mode() -> str:
    return str(_pose().get("mode", "balanced"))


def get_checkpoint_urls() -> dict[str, str]:
    urls = _models().get("download_urls", {})
    return {str(k): str(v) for k, v in urls.items()}


def get_model_aliases() -> dict[str, list[str]]:
    raw = _models().get("aliases", {})
    return {str(name): [str(alias) for alias in aliases] for name, aliases in raw.items()}
