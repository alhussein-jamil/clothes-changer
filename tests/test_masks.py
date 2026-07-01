import numpy as np
from PIL import Image

from outfit_studio.ui.masks import (
    apply_masks_to_editor,
    editor_mask_reset,
    letterbox_to_editor_canvas,
    parse_editor_masks,
    resolve_masks_for_generate,
)
from outfit_studio.ui.theme import CLOTHES_COLOR, EDITOR_CANVAS_SIZE, PERSON_COLOR


def test_letterbox_to_editor_canvas_centers_portrait():
    tall = Image.new("RGB", (800, 1200), color=(128, 64, 32))
    canvas = letterbox_to_editor_canvas(tall)
    assert canvas.size == EDITOR_CANVAS_SIZE
    arr = np.array(canvas.convert("RGB"))
    assert arr[500, 500, 0] > 0
    assert arr[10, 10, 0] == 0


def test_letterbox_unletterbox_roundtrip():
    from outfit_studio.ui.masks import letterbox_masks, unletterbox_masks

    src_size = (800, 600)
    person = np.zeros((600, 800), dtype=np.uint8)
    clothes = np.zeros((600, 800), dtype=np.uint8)
    clothes[100:300, 200:500] = 1

    lb_person, lb_clothes = letterbox_masks(person, clothes, src_size)
    back_person, back_clothes = unletterbox_masks(lb_person, lb_clothes, src_size)

    assert back_person.shape == person.shape
    assert int(back_clothes.sum()) > 0
    assert back_clothes[150, 350] == 1


def test_resolve_masks_prefers_cached_segment_masks_when_layers_stripped():
    """Avoid composite diff when ML masks are cached but Gradio drops layers."""
    from outfit_studio.utils.image import mask_overlay

    bg = Image.new("RGB", (64, 48), color=(100, 100, 100))
    person = np.zeros((48, 64), dtype=np.uint8)
    clothes = np.zeros((48, 64), dtype=np.uint8)
    person[10:30, 10:25] = 1
    clothes[15:25, 15:20] = 1

    editor = apply_masks_to_editor(bg, person, clothes)
    editor["layers"] = []
    editor["composite"] = mask_overlay(bg, person, clothes)

    pipeline_source = Image.new("RGB", (64, 48), color=(100, 100, 100))
    resolved_person, resolved_clothes = resolve_masks_for_generate(
        editor,
        (person, clothes),
        pipeline_source,
    )

    assert resolved_person is not None and resolved_person[20, 15] == 1
    assert resolved_clothes is not None and resolved_clothes[20, 18] == 1


def test_apply_and_parse_roundtrip():
    bg = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(100, 100, 100))
    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    person[100:200, 100:150] = 1
    clothes[250:350, 400:450] = 1

    editor = apply_masks_to_editor(bg, person, clothes)
    parsed_bg, parsed_person, parsed_clothes = parse_editor_masks(editor)

    assert parsed_bg is not None
    assert parsed_person is not None
    assert parsed_clothes is not None
    assert parsed_person[150, 120] == 1
    assert parsed_clothes[300, 420] == 1


def test_parse_masks_from_composite_when_layers_empty():
    """Fallback when Gradio sends composite but strips layers."""
    from outfit_studio.utils.image import mask_overlay

    bg = Image.new("RGB", (64, 48), color=(100, 100, 100))
    person = np.zeros((48, 64), dtype=np.uint8)
    clothes = np.zeros((48, 64), dtype=np.uint8)
    person[10:30, 10:25] = 1
    clothes[15:25, 15:20] = 1

    editor = apply_masks_to_editor(bg, person, clothes)
    editor["layers"] = []
    editor["composite"] = mask_overlay(bg, person, clothes)

    _, parsed_person, parsed_clothes = parse_editor_masks(editor)
    assert parsed_person is not None and parsed_person.sum() > 0
    assert parsed_clothes is not None and parsed_clothes.sum() > 0


def test_load_editor_clean_image_ignores_composite_overlay():
    from outfit_studio.ui.masks import load_editor_clean_image
    from outfit_studio.utils.image import mask_overlay

    bg = Image.new("RGB", (64, 48), color=(100, 100, 100))
    person = np.zeros((48, 64), dtype=np.uint8)
    clothes = np.zeros((48, 64), dtype=np.uint8)
    person[10:30, 10:25] = 1
    overlay = mask_overlay(bg, person, clothes)

    assert load_editor_clean_image({"background": None, "layers": [], "composite": overlay}) is None
    assert load_editor_clean_image({"background": bg, "layers": [], "composite": overlay}).size == (
        64,
        48,
    )


def test_parse_editor_uses_composite_when_background_missing(tmp_path):
    img_path = tmp_path / "input.png"
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(img_path)

    bg, person, clothes = parse_editor_masks(
        {"background": None, "layers": [], "composite": str(img_path)}
    )
    assert bg is not None
    assert bg.size == (40, 30)
    assert person is not None and person.sum() == 0
    assert clothes is not None and clothes.sum() == 0


def test_parse_empty_editor():
    bg, person, clothes = parse_editor_masks(None)
    assert bg is None
    assert person is None
    assert clothes is None


def test_parse_editor_with_filepath_background(tmp_path):
    img_path = tmp_path / "input.png"
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(img_path)

    bg, person, clothes = parse_editor_masks(
        {"background": str(img_path), "layers": [], "composite": None}
    )
    assert bg is not None
    assert bg.size == (40, 30)
    assert person is not None and person.sum() == 0
    assert clothes is not None and clothes.sum() == 0


def test_parse_brush_stroke_on_layer():
    bg = Image.new("RGBA", (64, 48), color=(100, 100, 100, 255))
    layer = Image.new("RGBA", (64, 48), color=(0, 0, 0, 0))
    layer.putpixel((20, 20), (0, 255, 0, 100))
    layer.putpixel((30, 25), (255, 0, 0, 100))

    _, person, clothes = parse_editor_masks(
        {"background": bg, "layers": [layer], "composite": None}
    )
    assert person is not None and person[25, 30] == 1
    assert clothes is not None and clothes[20, 20] == 1


def test_apply_masks_replaces_existing_layers():
    """Re-segment must replace layers, not append to prior mask strokes."""
    bg = Image.new("RGB", EDITOR_CANVAS_SIZE)
    stale = np.zeros((*EDITOR_CANVAS_SIZE[::-1], 4), dtype=np.uint8)
    stale[0:80, 0:80] = PERSON_COLOR
    editor = {
        "background": bg,
        "layers": [stale, stale.copy()],
        "composite": None,
    }
    person = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes = np.zeros(EDITOR_CANVAS_SIZE[::-1], dtype=np.uint8)
    clothes[100:200, 100:200] = 1

    result = apply_masks_to_editor(bg, person, clothes, editor=editor)

    assert len(result["layers"]) == 1
    assert isinstance(result["layers"][0], Image.Image)
    layer = np.array(result["layers"][0])
    assert layer[50, 50, 0] == 0
    assert layer[150, 150, 1] == CLOTHES_COLOR[1]
    assert result["composite"] is not None


def test_editor_mask_reset_clears_stacked_layers():
    clean = Image.new("RGB", EDITOR_CANVAS_SIZE, color=(10, 20, 30))
    stale = Image.new("RGBA", EDITOR_CANVAS_SIZE, color=(0, 0, 0, 0))
    stale.putpixel((50, 50), CLOTHES_COLOR)
    editor = {
        "background": clean.convert("RGBA"),
        "layers": [stale, stale.copy(), stale.copy()],
        "composite": None,
    }

    reset = editor_mask_reset(editor, clean)

    assert reset["layers"] == []
    assert reset["background"].size == EDITOR_CANVAS_SIZE
    assert np.array(reset["composite"])[50, 50].tolist() == [10, 20, 30]


def test_apply_masks_background_matches_layer_size():
    bg = Image.new("RGBA", (32, 32), color=(10, 20, 30, 255))
    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    clothes[5:10, 5:10] = 1

    result = apply_masks_to_editor(bg.convert("RGB"), person, clothes)
    assert result["background"].size == (32, 32)
    assert np.array(result["layers"][0]).shape == (32, 32, 4)
    assert result["composite"] is not None
    assert isinstance(result["layers"][0], Image.Image)


def test_apply_masks_without_editor_uses_native_size():
    bg = Image.new("RGB", (800, 600), color=(50, 60, 70))
    person = np.zeros((600, 800), dtype=np.uint8)
    clothes = np.zeros((600, 800), dtype=np.uint8)
    clothes[100:200, 100:200] = 1

    result = apply_masks_to_editor(bg, person, clothes)
    assert result["background"].size == (800, 600)
    assert np.array(result["layers"][0]).shape == (600, 800, 4)


def test_parse_editor_with_filedata_background(tmp_path):
    img_path = tmp_path / "input.png"
    Image.new("RGB", (20, 20), color=(255, 0, 0)).save(img_path)

    bg, _, _ = parse_editor_masks(
        {
            "background": {"path": str(img_path), "orig_name": "input.png"},
            "layers": [],
            "composite": None,
        }
    )
    assert bg is not None
    assert bg.size == (20, 20)
