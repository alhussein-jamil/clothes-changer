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


def _brand_name_html(name: str) -> str:
    """Split trailing word as muted weight — Game Studio style (Outfit <b>Studio</b>)."""
    parts = name.strip().split()
    if len(parts) >= 2:
        head = html.escape(" ".join(parts[:-1]))
        tail = html.escape(parts[-1])
        return f"{head} <b>{tail}</b>"
    return html.escape(name)


def build_header_html(settings: Settings) -> str:
    """Studio chrome header: mark + brand + tagline."""
    logo = settings.resolved_logo_path
    name = get_app_name()
    tagline = get_tagline().strip()
    tagline_html = f'<p class="app-header-tagline">{html.escape(tagline)}</p>' if tagline else ""
    mark = (
        f'<img class="app-mark" src="/file={logo}" '
        f'width="{UI.MARK_SIZE_PX}" height="{UI.MARK_SIZE_PX}" '
        f'alt="" />'
    )

    return "\n".join(
        [
            '<div class="app-header">',
            '<div class="app-topbar">',
            '<div class="app-brand">',
            mark,
            f'<p class="app-brand-name">{_brand_name_html(name)}</p>',
            "</div>",
            '<p class="app-eyebrow">Local · Inpaint</p>',
            "</div>",
            '<div class="app-header-body">',
            # Keep class for tests / a11y title hook; visually replaced by brand-name.
            f'<div class="app-header-title"><h1>{html.escape(name)}</h1></div>',
            tagline_html,
            "</div>",
            "</div>",
        ]
    )
