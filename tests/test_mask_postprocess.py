import numpy as np

from outfit_studio.ml.mask_postprocess import refine_segmentation_masks


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
    assert person_out.sum() >= person.sum()


def test_refine_segmentation_masks_drops_small_components():
    person = np.ones((30, 30), dtype=np.uint8)
    clothes = np.zeros((30, 30), dtype=np.uint8)
    clothes[5, 5] = 1  # single-pixel speckle
    clothes[10:20, 10:20] = 1

    _, clothes_out = refine_segmentation_masks(person, clothes, min_component_area=4)

    assert clothes[5, 5] == 1
    assert clothes_out[5, 5] == 0
    assert clothes_out[10:20, 10:20].sum() > 0
