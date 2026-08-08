"""UI layout/CSS guards for ImageEditor preview sizing."""

from __future__ import annotations

import inspect

from outfit_studio.ui.tabs import layout
from outfit_studio.ui.theme import CUSTOM_CSS, UI


def test_editor_has_explicit_height():
    src = inspect.getsource(layout.build_ui)
    assert "height=UI.EDITOR_HEIGHT_PX" in src
    assert UI.EDITOR_HEIGHT_PX >= 480


def test_custom_css_keeps_upload_overlay_transparent():
    # Opaque .upload-container fill hides Gradio Pixi (blank preview / bottom sliver).
    assert "#studio-input-editor .upload-container" in CUSTOM_CSS
    assert "background: transparent !important" in CUSTOM_CSS
    assert ".image-container, .image-frame, .upload-container" not in CUSTOM_CSS
    assert "animation: os-rise" not in CUSTOM_CSS


def test_generate_row_does_not_use_equal_height():
    src = inspect.getsource(layout.build_ui)
    assert 'elem_id="studio-generate-row"' in src
    assert 'elem_id="studio-input-editor"' in src
    assert "Row(equal_height=True)" not in src
    assert "gr.Row(equal_height=True)" not in src
