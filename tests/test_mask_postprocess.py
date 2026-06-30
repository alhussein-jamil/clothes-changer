import numpy as np
from PIL import Image
from scipy import ndimage

from outfit_studio.ml.mask_postprocess import (
    normalize_nested_masks,
    refine_segmentation_masks,
)
from outfit_studio.utils.image import mask_overlay


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

    person_out, clothes_out = refine_segmentation_masks(person, clothes, min_component_area=0)

    ring_after = (person_out > 0) & ~(clothes_out > 0)
    ring_after &= ndimage.binary_dilation(clothes_out > 0)
    assert ring_after.sum() == 0

    overlay = np.array(
        mask_overlay(Image.new("RGB", (100, 100), color=(128, 128, 128)), person_out, clothes_out)
    )
    red_halo = (
        (overlay[:, :, 0] > overlay[:, :, 1] + 20)
        & (overlay[:, :, 3] > 0)
        & ndimage.binary_dilation(clothes_out > 0)
    )
    assert red_halo.sum() == 0


def test_normalize_nested_masks_skips_disjoint_editor_masks():
    person = np.zeros((50, 50), dtype=np.uint8)
    clothes = np.zeros((50, 50), dtype=np.uint8)
    person[5:15, 5:15] = 1
    clothes[30:40, 30:40] = 1

    person_out, clothes_out = normalize_nested_masks(person, clothes)
    assert person_out.sum() == person.sum()
    assert clothes_out.sum() == clothes.sum()


def test_normalize_nested_masks_refines_segmentation_output():
    person = np.zeros((100, 100), dtype=np.uint8)
    clothes = np.zeros((100, 100), dtype=np.uint8)
    person[20:80, 20:80] = 1
    clothes[22:78, 22:78] = 1

    _, clothes_out = normalize_nested_masks(person, clothes)
    ring = (person > 0) & ~(clothes_out > 0)
    ring &= ndimage.binary_dilation(clothes_out > 0)
    assert ring.sum() == 0
