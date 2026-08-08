import numpy as np
from PIL import Image
from scipy import ndimage

from outfit_studio.ml.mask_postprocess import refine_segmentation_masks
from outfit_studio.utils.image import mask_overlay


def test_refine_fills_enclosed_hole_in_clothes():
    """Pants fly / mid-garment gaps should be patched closed."""
    person = np.ones((60, 40), dtype=np.uint8)
    clothes = np.ones((60, 40), dtype=np.uint8)
    clothes[25:40, 18:22] = 0  # enclosed vertical hole

    person_out, clothes_out = refine_segmentation_masks(
        person,
        clothes,
        min_component_area=0,
        clothes_smooth_px=0,
        clothes_edge_grow_px=0,
    )

    assert clothes_out[30, 20] == 1
    assert clothes_out[25:40, 18:22].min() == 1
    np.testing.assert_array_equal(person_out, person)


def test_refine_does_not_fill_person_holes():
    person = np.ones((40, 40), dtype=np.uint8)
    person[15:25, 15:25] = 0
    clothes = np.zeros((40, 40), dtype=np.uint8)
    clothes[5:35, 5:35] = 1
    clothes[15:25, 15:25] = 0

    person_out, clothes_out = refine_segmentation_masks(
        person,
        clothes,
        min_component_area=0,
        clothes_smooth_px=0,
        clothes_edge_grow_px=0,
    )

    assert person_out[20, 20] == 0
    assert clothes_out[20, 20] == 0


def test_refine_segmentation_masks_constrains_clothes_to_person():
    person = np.zeros((30, 30), dtype=np.uint8)
    clothes = np.zeros((30, 30), dtype=np.uint8)
    person[5:25, 5:25] = 1
    clothes[10:20, 10:20] = 1
    clothes[0:5, 0:5] = 1  # outside person

    person_out, clothes_out = refine_segmentation_masks(
        person,
        clothes,
        min_component_area=0,
    )

    assert clothes_out[0:5, 0:5].sum() == 0
    assert clothes_out[10:20, 10:20].sum() > 0
    assert person_out.sum() == person.sum()


def test_refine_segmentation_masks_drops_small_components():
    person = np.ones((30, 30), dtype=np.uint8)
    clothes = np.zeros((30, 30), dtype=np.uint8)
    clothes[5, 5] = 1  # single-pixel speckle
    clothes[10:20, 10:20] = 1

    _, clothes_out = refine_segmentation_masks(person, clothes, min_component_area=4)

    assert clothes[5, 5] == 1
    assert clothes_out[5, 5] == 0
    assert clothes_out[10:20, 10:20].sum() > 0


def test_refine_segmentation_masks_closes_clothes_edge_ring():
    """Parser-style undershoot leaves a person-only ring; grow should remove it."""
    person = np.zeros((100, 100), dtype=np.uint8)
    clothes = np.zeros((100, 100), dtype=np.uint8)
    person[20:80, 20:80] = 1
    clothes[22:78, 22:78] = 1

    person_only_ring = (person > 0) & ~(clothes > 0)
    ring_before = person_only_ring & ndimage.binary_dilation(clothes > 0)
    assert ring_before.sum() > 0

    person_out, clothes_out = refine_segmentation_masks(
        person, clothes, min_component_area=0, clothes_smooth_px=0
    )

    ring_after = (person_out > 0) & ~(clothes_out > 0)
    ring_after &= ndimage.binary_dilation(clothes_out > 0)
    assert ring_after.sum() == 0

    overlay = np.array(
        mask_overlay(Image.new("RGB", (100, 100), color=(128, 128, 128)), person_out, clothes_out)
    )
    # Interior of clothes must stay green-dominant (no person red under spray).
    core = clothes_out[35:65, 35:65].astype(bool)
    core_pixels = overlay[35:65, 35:65][core]
    assert core_pixels[:, 1].mean() > core_pixels[:, 0].mean() + 10
