"""Branded copy, prompts, and model defaults from YAML (local overrides default)."""

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


def get_app_name() -> str:
    return str(get_content_config().get("app", {}).get("name", "Clothes Changer"))


def get_tagline() -> str:
    return str(get_content_config().get("app", {}).get("tagline", ""))


def get_default_prompt() -> str:
    return str(get_content_config().get("prompts", {}).get("default", ""))


def get_default_negative_prompt() -> str:
    return str(get_content_config().get("prompts", {}).get("negative", ""))


def get_title_html() -> str:
    return str(get_content_config().get("ui", {}).get("title_html", "")).strip()


def get_default_inpaint_model() -> str:
    return str(
        get_content_config()
        .get("models", {})
        .get("default_inpaint", "realisticVisionV60B1_v51HyperInpaintVAE.safetensors")
    )


def get_checkpoint_urls() -> dict[str, str]:
    urls = get_content_config().get("models", {}).get("download_urls", {})
    return {str(k): str(v) for k, v in urls.items()}
