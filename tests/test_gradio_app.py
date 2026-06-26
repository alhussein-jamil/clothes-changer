from unittest.mock import patch

import gradio as gr
import numpy as np
from PIL import Image

from clothes_changer.ui.constants import EDITOR_CANVAS_SIZE
from clothes_changer.ui.gradio_app import GradioApp
from clothes_changer.ui.masks import apply_masks_to_editor


def test_authenticate(db):
    app = GradioApp(db=db)
    db.register_user("user1", "password123", credits=5)
    assert app.authenticate("user1", "password123")
    assert not app.authenticate("user1", "wrong")


def _editor_value(update: dict) -> dict:
    return update.get("value", update)


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_load_history_and_segment(mock_get_segmentor, db, tmp_path):
    app = GradioApp(db=db)
    img_path = tmp_path / "admin_test.png"
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(img_path)

    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    clothes[10:20, 10:20] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros((32, 32)),
        person,
        clothes,
    )

    evt = gr.SelectData(
        target=None,
        data={"index": 0, "value": str(img_path), "selected": True},
    )
    editor_update, clean, key, skip, _ = app.load_history_and_segment(evt, None, None)
    value = _editor_value(editor_update)
    assert np.array(value["layers"][0])[15, 15, 1] > 50  # clothes mask visible in layer
    assert clean.size == (32, 32)
    assert key is not None
    assert skip is True


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_segment_after_example(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(200, 100, 50))

    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)  # HxW
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes[50:150, 50:150] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros(EDITOR_CANVAS_SIZE[::-1]),
        person,
        clothes,
    )

    editor = apply_masks_to_editor(
        bg,
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
    )
    editor_update, clean, key, skip, _ = app.segment_after_example(editor, None, None)
    value = _editor_value(editor_update)
    assert np.array(value["layers"][0])[100, 100, 1] > 50
    assert clean is not None
    assert key is not None
    assert skip is True


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_segment_if_unmasked_runs_when_no_masks(mock_get_segmentor, db):
    app = GradioApp(db=db)
    Image.new("RGB", (32, 32))
    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes[50:150, 50:150] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros(EDITOR_CANVAS_SIZE[::-1]),
        person,
        clothes,
    )

    editor = apply_masks_to_editor(
        Image.new("RGB", EDITOR_CANVAS_SIZE),
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
    )
    editor_value, clean = app.segment_if_unmasked(editor)
    value = editor_value.get("value", editor_value)
    assert np.array(value["layers"][0])[100, 100, 1] > 50
    assert clean is not None


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_segment_if_unmasked_skips_existing_masks(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg = Image.new("RGB", (32, 32))
    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    clothes[5:15, 5:15] = 1
    editor = apply_masks_to_editor(bg, person, clothes)
    editor_value, clean = app.segment_if_unmasked(editor)
    assert editor_value == gr.update()
    assert clean is not None
    mock_get_segmentor.assert_not_called()


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_segment(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg = Image.new("RGB", (32, 32))
    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    clothes[5:15, 5:15] = 1
    mock_get_segmentor.return_value.segment.return_value = (np.zeros((32, 32)), person, clothes)

    editor = apply_masks_to_editor(bg, person, clothes)
    editor_value, clean = app.segment(editor)
    assert editor_value is not None
    assert "layers" in editor_value
    assert clean is not None


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_run_segmentation_uses_full_res_editor_background(mock_get_segmentor, db):
    """Masks must match Gradio's native background size, not a letterboxed canvas."""
    app = GradioApp(db=db)
    full_bg = Image.new("RGB", (800, 600), color=(100, 50, 25))
    person = np.zeros((600, 800), dtype=np.uint8)
    clothes = np.zeros((600, 800), dtype=np.uint8)
    clothes[100:200, 100:200] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros((600, 800)),
        person,
        clothes,
    )

    editor = {"background": full_bg, "layers": [], "composite": None}
    result = app._run_segmentation(editor)

    seg_call = mock_get_segmentor.return_value.segment.call_args[0][0]
    assert seg_call.size == (800, 600)
    assert result.pipeline_clean.size == (800, 600)
    assert result.editor_value["background"].size == (800, 600)
    assert np.array(result.editor_value["layers"][0]).shape == (600, 800, 4)
    assert result.editor_value["composite"] is not None
    assert int(np.array(result.editor_value["layers"][0])[:, :, 1].sum()) > 0


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_resegment_recovers_from_segment_key(mock_get_segmentor, db, tmp_path):
    app = GradioApp(db=db)
    img_path = tmp_path / "redo.png"
    Image.new("RGB", (64, 64), color=(40, 80, 120)).save(img_path)
    key = f"path:{img_path.resolve()}"

    person = np.zeros((64, 64), dtype=np.uint8)
    clothes = np.zeros((64, 64), dtype=np.uint8)
    clothes[20:40, 20:40] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros((64, 64)),
        person,
        clothes,
    )

    empty_editor = {"background": None, "layers": [], "composite": None}
    cleared, clean, out_key, skip_prepare, _ = app.resegment_prepare(
        empty_editor, None, key, None, None
    )
    cleared_value = _editor_value(cleared)
    assert cleared_value["layers"] == []
    editor_update, clean, out_key, skip, _ = app.resegment(
        cleared_value, clean, out_key, None, None
    )
    value = _editor_value(editor_update)
    assert np.array(value["layers"][0])[30, 30, 1] > 50
    assert skip_prepare is True
    assert skip is True
    assert clean is not None
    assert out_key is not None


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_prepare_upload_segment_preserves_clean_source_on_empty_editor(mock_get_segmentor, db):
    app = GradioApp(db=db)
    clean = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(1, 2, 3))
    empty_editor = {"background": None, "layers": [], "composite": None}
    pending, out_clean, key, skip, _ = app.prepare_upload_segment(
        empty_editor, "path:/tmp/x.png", clean, False, None, None
    )
    assert pending == gr.skip()
    assert out_clean is clean
    assert key == "path:/tmp/x.png"
    mock_get_segmentor.assert_not_called()


def test_sync_clean_source_never_overwrites_pristine(db):
    app = GradioApp(db=db)
    from clothes_changer.utils.image import mask_overlay

    pristine = Image.new("RGB", (64, 48), color=(100, 100, 100))
    person = np.zeros((48, 64), dtype=np.uint8)
    person[10:30, 10:25] = 1
    overlay_bg = mask_overlay(pristine, person, np.zeros_like(person))

    editor = {
        "background": overlay_bg,
        "layers": [],
        "composite": overlay_bg,
    }
    out = app.sync_clean_source(editor, pristine, None)
    assert out is pristine


def test_pipeline_source_prefers_clean_source_over_editor(db, tmp_path):
    app = GradioApp(db=db)
    pristine = Image.new("RGB", (40, 30), color=(10, 20, 30))
    contaminated = Image.new("RGB", (40, 30), color=(255, 0, 0))
    editor = {"background": contaminated, "layers": [], "composite": None}
    assert app._pipeline_source(editor, pristine, None).getpixel((0, 0)) == (10, 20, 30)

    img_path = tmp_path / "src.png"
    pristine.save(img_path)
    key = f"path:{img_path.resolve()}"
    assert app._pipeline_source(None, None, key).size == pristine.size


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_prepare_upload_segment_retries_after_empty_masks(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(10, 20, 30))
    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes[80:200, 80:200] = 1
    mock_get_segmentor.return_value.segment.side_effect = [
        (
            np.zeros(EDITOR_CANVAS_SIZE[::-1]),
            np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
            np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
        ),
        (np.zeros(EDITOR_CANVAS_SIZE[::-1]), person, clothes),
    ]

    editor = apply_masks_to_editor(
        bg,
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
    )
    pending1, clean, key1, _, _ = app.prepare_upload_segment(editor, None, None, False, None, None)
    assert pending1 == gr.skip()
    assert key1 is not None
    assert mock_get_segmentor.return_value.segment.call_count == 1

    pending2, _, key2, skip2, _ = app.prepare_upload_segment(editor, key1, clean, False, None, None)
    assert pending2 is not None
    assert key2 is not None
    assert skip2 is True
    assert mock_get_segmentor.return_value.segment.call_count == 2


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_prepare_upload_segment_skips_repeat(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg = Image.new("RGB", (32, 32), color=(10, 20, 30))
    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    clothes[5:15, 5:15] = 1
    mock_get_segmentor.return_value.segment.return_value = (np.zeros((32, 32)), person, clothes)

    editor = apply_masks_to_editor(
        bg,
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.uint8),
    )
    pending, clean, key, skip, _ = app.prepare_upload_segment(editor, None, None, False, None, None)
    assert pending is not None
    assert clean is not None
    assert key is not None
    assert skip is True
    assert mock_get_segmentor.return_value.segment.call_count == 1
    assert np.array(_editor_value(pending)["layers"][0]).shape == (32, 32, 4)

    masked_editor = apply_masks_to_editor(bg, person, clothes)
    pending2, _, same_key, skip2, _ = app.prepare_upload_segment(
        masked_editor, key, clean, False, None, None
    )
    assert pending2 == gr.skip()
    assert same_key == key
    assert skip2 is False
    assert mock_get_segmentor.return_value.segment.call_count == 1


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_prepare_upload_segment_resegments_when_masks_stale(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg_a = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(10, 20, 30))
    bg_b = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(40, 50, 60))
    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes[80:200, 80:200] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros(EDITOR_CANVAS_SIZE[::-1]),
        person,
        clothes,
    )

    editor_a = apply_masks_to_editor(
        bg_a,
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
        np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8),
    )
    pending, clean, key_a, _, _ = app.prepare_upload_segment(
        editor_a, None, None, False, None, None
    )
    assert pending is not None
    assert mock_get_segmentor.return_value.segment.call_count == 1

    masked_a = apply_masks_to_editor(bg_a, person, clothes)
    pending2, _, same_key, _, _ = app.prepare_upload_segment(
        masked_a, key_a, clean, False, None, None
    )
    assert pending2 == gr.skip()
    assert same_key == key_a
    assert mock_get_segmentor.return_value.segment.call_count == 1

    masked_b = apply_masks_to_editor(bg_b, person, clothes)
    pending3, _, key_b, _, _ = app.prepare_upload_segment(masked_b, key_a, clean, False, None, None)
    assert pending3 is not None
    assert key_b != key_a
    assert mock_get_segmentor.return_value.segment.call_count == 2


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_load_example_after_select(mock_get_segmentor, db, tmp_path):
    app = GradioApp(db=db)
    img_path = tmp_path / "example.png"
    Image.new("RGB", (48, 48), color=(200, 100, 50)).save(img_path)
    app.examples = [str(img_path)]

    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes[80:200, 80:200] = 1
    mock_get_segmentor.return_value.segment.return_value = (
        np.zeros(EDITOR_CANVAS_SIZE[::-1]),
        person,
        clothes,
    )

    editor = {
        "background": Image.new("RGBA", EDITOR_CANVAS_SIZE, color=(50, 50, 50, 255)),
        "layers": [],
        "composite": None,
    }
    editor_update, clean, key, skip, _ = app.load_example_after_select(editor, 0, None, None)
    value = _editor_value(editor_update)
    assert np.array(value["layers"][0])[150, 150, 1] > 50
    assert clean is not None
    assert key.startswith("path:")
    assert skip is True


@patch("clothes_changer.ml.segmentor.get_segmentor")
def test_prepare_upload_segment_skips_programmatic_load(mock_get_segmentor, db):
    app = GradioApp(db=db)
    bg = Image.new("RGB", (32, 32), color=(10, 20, 30))
    editor = apply_masks_to_editor(
        bg, np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8)
    )
    pending, clean, key, skip, _ = app.prepare_upload_segment(
        editor, "path:/tmp/x.png", bg, True, None, None
    )
    assert pending == gr.skip()
    assert clean is bg
    assert key == "path:/tmp/x.png"
    assert skip is False
    mock_get_segmentor.assert_not_called()
