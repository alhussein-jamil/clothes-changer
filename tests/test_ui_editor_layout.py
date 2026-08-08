"""UI layout/CSS guards for ImageEditor preview sizing."""

from __future__ import annotations

import inspect

from outfit_studio.ui.tabs import layout
from outfit_studio.ui.theme import CUSTOM_CSS


def test_custom_css_keeps_image_editor_min_height():
    assert "min-height: 420px" in CUSTOM_CSS
    # Transform rise on .image-container breaks Gradio canvas layout.
    assert "animation: os-rise" not in CUSTOM_CSS


def test_generate_row_does_not_use_equal_height():
    src = inspect.getsource(layout.build_ui)
    assert 'elem_id="studio-generate-row"' in src
    assert "Row(equal_height=True)" not in src
    assert "gr.Row(equal_height=True)" not in src
