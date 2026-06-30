"""Page header and shared segmentation result types."""

from __future__ import annotations

import html
from typing import NamedTuple

import numpy as np
from PIL import Image

from outfit_studio.config import Settings
from outfit_studio.content_config import get_app_name, get_tagline
from outfit_studio.ui.theme import UI


class SegmentationResult(NamedTuple):
    """Return value from segmentation handlers."""

    editor_value: dict
    pipeline_clean: Image.Image
    person: np.ndarray
    clothes: np.ndarray
    debug_session_dir: str | None


def _header_title_block(name: str) -> str:
    tagline = get_tagline().strip()
    tagline_html = f'<p class="app-header-tagline">{html.escape(tagline)}</p>' if tagline else ""
    return f'<div class="app-header-title"><h1>{html.escape(name)}</h1>{tagline_html}</div>'


def _logo_image_style() -> str:
    return (
        f"max-width:min({UI.LOGO_MAX_WIDTH_PX}px,100%);"
        f"height:auto;max-height:{UI.LOGO_MAX_HEIGHT_PX}px"
    )


def build_header_html(settings: Settings) -> str:
    """Page header with logo and app name."""
    logo = settings.resolved_logo_path
    name = get_app_name()
    logo_style = _logo_image_style()

    return "\n".join(
        [
            '<div class="app-header">',
            '<div class="app-header-brand">',
            '<div class="app-header-logo-wrap">',
            f'<img src="/file={logo}" alt="{html.escape(name)}" style="{logo_style}" />',
            "</div>",
            _header_title_block(name),
            "</div>",
            "</div>",
        ]
    )
