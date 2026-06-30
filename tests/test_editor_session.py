import numpy as np
from PIL import Image

from outfit_studio.ui.editor_session import (
    EditorSession,
    UploadSegmentAction,
    evaluate_upload_segment,
    resolve_clean_on_upload,
)
from outfit_studio.ui.masks import apply_masks_to_editor, background_key_from_image
from outfit_studio.ui.theme import EDITOR_CANVAS_SIZE
from outfit_studio.utils.image import mask_overlay


def test_session_cleared_fields():
    assert EditorSession().cleared_fields() == (None, None, False, None)


def test_session_fields_after_programmatic_push():
    clean = Image.new("RGB", (8, 8))
    session = EditorSession.from_fields(None, None, None, False)
    assert session.fields_after_programmatic_push(clean, "key:1", "/tmp/debug") == (
        clean,
        "key:1",
        True,
        "/tmp/debug",
    )


def test_resolve_clean_preserves_pristine_during_mask_editing():
    pristine = Image.new("RGB", (64, 48), color=(100, 100, 100))
    person = np.zeros((48, 64), dtype=np.uint8)
    person[10:30, 10:25] = 1
    overlay_bg = mask_overlay(pristine, person, np.zeros_like(person))
    editor = {"background": overlay_bg, "layers": [], "composite": overlay_bg}
    session = EditorSession.from_fields(pristine, None, None, False)
    assert resolve_clean_on_upload(editor, session) is pristine


def test_resolve_clean_refreshes_when_image_changes():
    old_clean = Image.new("RGB", (64, 48), color=(10, 20, 30))
    new_bg = Image.new("RGB", (64, 48), color=(40, 50, 60))
    old_key = background_key_from_image(old_clean)
    editor = {"background": new_bg, "layers": [], "composite": new_bg}
    session = EditorSession.from_fields(old_clean, old_key, None, False)
    out = resolve_clean_on_upload(editor, session)
    assert out.getpixel((0, 0)) == (40, 50, 60)


def test_evaluate_upload_segment_no_background():
    session = EditorSession()
    action, clean, key = evaluate_upload_segment(
        {"background": None, "layers": [], "composite": None}, session
    )
    assert action is UploadSegmentAction.SKIP_NO_BACKGROUND
    assert clean is None
    assert key is None


def test_evaluate_upload_segment_programmatic_skip():
    bg = Image.new("RGB", (32, 32), color=(10, 20, 30))
    key = background_key_from_image(bg)
    editor = apply_masks_to_editor(
        bg, np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8)
    )
    session = EditorSession.from_fields(bg, key, None, suppress_upload_hook=True)
    action, clean, out_key = evaluate_upload_segment(editor, session)
    assert action is UploadSegmentAction.SKIP_PROGRAMMATIC
    assert clean is bg
    assert out_key == key


def test_evaluate_upload_segment_new_image_ignores_stale_suppress_flag():
    bg_old = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(10, 20, 30))
    bg_new = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(40, 50, 60))
    old_key = background_key_from_image(bg_old)
    empty = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    editor = apply_masks_to_editor(bg_new, empty, empty)
    session = EditorSession.from_fields(bg_old, old_key, None, suppress_upload_hook=True)
    action, clean, key = evaluate_upload_segment(editor, session)
    assert action is UploadSegmentAction.SEGMENT
    assert key != old_key
    assert clean.getpixel((0, 0)) == (40, 50, 60)


def test_evaluate_upload_segment_skips_when_masks_present():
    bg = Image.new("RGB", (32, 32), color=(10, 20, 30))
    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    clothes[5:15, 5:15] = 1
    key = background_key_from_image(bg)
    editor = apply_masks_to_editor(bg, person, clothes)
    session = EditorSession.from_fields(bg, key, None, False)
    action, clean, out_key = evaluate_upload_segment(editor, session)
    assert action is UploadSegmentAction.SKIP_MASKS_PRESENT
    assert clean is bg
    assert out_key == key
