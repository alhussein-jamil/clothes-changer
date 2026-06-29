import numpy as np
from PIL import Image

from clothes_changer.ml.segmentor import CLOTHES_CATEGORIES, PERSON_CATEGORIES
from clothes_changer.utils.image import (
    blend_images_with_enhancements,
    get_bounding_box,
    inpaint_mask_from_clothes,
    mask_overlay,
    prepare_instance_masks,
    resize_max,
    separate_instances,
)


def test_segformer_category_constants():
    assert 11 in PERSON_CATEGORIES  # sunglasses/glasses → person, not clothes
    assert 4 in CLOTHES_CATEGORIES


def test_resize_max():
    img = Image.new("RGB", (2000, 1000))
    out = resize_max(img, 1024)
    assert max(out.size) == 1024


def test_bounding_box_and_instances():
    person = np.zeros((50, 50), dtype=np.uint8)
    clothes = np.zeros((50, 50), dtype=np.uint8)
    person[10:30, 10:30] = 1
    clothes[15:25, 15:25] = 1
    bbox = get_bounding_box(person | clothes)
    assert bbox[0] == 10
    instances = separate_instances(person, clothes)
    assert len(instances) == 1


def test_prepare_instance_masks_per_bbox():
    person = np.zeros((100, 100), dtype=np.uint8)
    clothes = np.zeros((100, 100), dtype=np.uint8)
    person[10:50, 10:50] = 1
    clothes[20:40, 20:40] = 1
    bboxes = np.array([[5, 5, 55, 55]], dtype=np.float32)
    instances = prepare_instance_masks(person, clothes, bboxes)
    assert len(instances) == 1
    assert instances[0][0].sum() > 0


def test_clip_bbox_when_mask_wider_than_image():
    from clothes_changer.utils.image import crop_square, pad_bbox

    bbox = (100, 400, 400, 700)
    clipped = pad_bbox(bbox, (576, 384))
    assert clipped[1] < clipped[3]
    assert clipped[0] < clipped[2]

    img = Image.new("RGB", (384, 576))
    mask = np.zeros((576, 384), dtype=np.uint8)
    mask[100:400, 50:350] = 1
    crop_square(img, mask, clipped)


def test_align_masks_resizes():
    from clothes_changer.utils.image import align_masks

    person = np.ones((100, 200), dtype=np.uint8)
    clothes = np.zeros((100, 200), dtype=np.uint8)
    p, c = align_masks(person, clothes, 50, 100)
    assert p.shape == (50, 100)
    assert c.shape == (50, 100)


def test_mask_overlay():
    img = Image.new("RGB", (20, 20), color=(128, 128, 128))
    person = np.zeros((20, 20), dtype=np.uint8)
    clothes = np.zeros((20, 20), dtype=np.uint8)
    clothes[5:10, 5:10] = 1
    overlay = mask_overlay(img, person, clothes)
    assert overlay.mode == "RGBA"


def test_blend_covers_grown_garment_edge():
    original = Image.new("RGB", (100, 100), color=(20, 20, 20))
    inpainted = Image.new("RGB", (100, 100), color=(220, 0, 0))

    clothes = Image.new("L", (100, 100), 0)
    clothes_np = np.array(clothes)
    clothes_np[20:80, 20:80] = 255
    clothes = Image.fromarray(clothes_np, mode="L")

    person = Image.new("L", (100, 100), 0)
    person_np = np.array(person)
    person_np[15:85, 15:85] = 255
    person = Image.fromarray(person_np, mode="L")

    result = blend_images_with_enhancements(original, inpainted, clothes, person)

    assert result.getpixel((50, 50))[:3] == (220, 0, 0)
    edge = result.getpixel((19, 50))[:3]
    assert edge != (20, 20, 20)
    assert edge[0] > edge[2]
    assert result.getpixel((10, 50))[:3] == (20, 20, 20)


def test_composite_crop_onto_respects_alpha():
    from clothes_changer.utils.image import composite_crop_onto

    full = Image.new("RGB", (50, 50), color=(0, 0, 255))
    patch = Image.new("RGBA", (20, 20), color=(255, 0, 0, 0))
    patch.putpixel((10, 10), (255, 0, 0, 255))
    out = composite_crop_onto(full, patch, 5, 5)
    assert out.getpixel((5, 5)) == (0, 0, 255)
    assert out.getpixel((15, 15)) == (255, 0, 0)


def test_inpaint_mask_from_clothes_expands_edges():
    clothes = np.zeros((80, 80), dtype=np.uint8)
    clothes[20:60, 20:60] = 1

    expanded = inpaint_mask_from_clothes(clothes)

    assert expanded.sum() > clothes.sum()
    assert expanded[19, 40] == 1


def test_generation_skips_instances_without_clothes(monkeypatch):
    from clothes_changer.ml.pipeline import GenerationPipeline

    source = Image.new("RGB", (200, 200), color=(20, 20, 20))
    person = np.zeros((200, 200), dtype=np.uint8)
    clothes = np.zeros((200, 200), dtype=np.uint8)
    clothes[50:150, 50:100] = 1
    person[50:150, 50:150] = 1

    calls: list[int] = []

    class DummyPose:
        @staticmethod
        def get_bboxes(image):
            return np.array(
                [[0, 0, 100, 200], [100, 0, 200, 200]],
                dtype=np.float32,
            )

        @staticmethod
        def unload():
            pass

    def fake_process(self, *args, **kwargs):
        calls.append(1)
        return Image.new("RGB", (50, 50), color=(30, 30, 30)), {
            "left": 50,
            "top": 50,
            "right": 100,
            "bottom": 100,
        }

    monkeypatch.setattr("clothes_changer.ml.pipeline.release_segmentation_gpu", lambda: None)
    monkeypatch.setattr("clothes_changer.ml.pipeline.free_cuda_cache", lambda: None)
    monkeypatch.setattr("clothes_changer.ml.pipeline.get_pose_estimator", lambda: DummyPose())
    monkeypatch.setattr(GenerationPipeline, "_process_single_mask", fake_process)
    monkeypatch.setattr(
        "clothes_changer.ml.pipeline.get_inpaint_engine",
        lambda: type("Engine", (), {"load": lambda *a, **k: None})(),
    )

    result, _, _ = GenerationPipeline().generate(
        source,
        person_mask=person,
        clothes_mask=clothes,
        prompt="prompt",
        negative_prompt="negative",
        username="test",
    )

    assert len(calls) == 1
    assert result.size == source.size


def test_generation_preserves_source_size(monkeypatch):
    from clothes_changer.ml.pipeline import GenerationPipeline

    source = Image.new("RGB", (1400, 900), color=(20, 20, 20))
    person = np.zeros((900, 1400), dtype=np.uint8)
    clothes = np.zeros((900, 1400), dtype=np.uint8)
    clothes[200:500, 300:700] = 1

    class DummyPose:
        @staticmethod
        def get_bboxes(image):
            return np.array([[0, 0, image.width, image.height]], dtype=np.float32)

        @staticmethod
        def unload():
            pass

    def fake_process(self, *args, **kwargs):
        return Image.new("RGB", (12, 12), color=(30, 30, 30)), {
            "left": 0,
            "top": 0,
            "right": 12,
            "bottom": 12,
        }

    monkeypatch.setattr("clothes_changer.ml.pipeline.release_segmentation_gpu", lambda: None)
    monkeypatch.setattr("clothes_changer.ml.pipeline.free_cuda_cache", lambda: None)
    monkeypatch.setattr("clothes_changer.ml.pipeline.get_pose_estimator", lambda: DummyPose())
    monkeypatch.setattr(
        "clothes_changer.ml.pipeline.prepare_instance_masks",
        lambda person_mask, clothes_mask, bboxes: [(person_mask, clothes_mask)],
    )
    monkeypatch.setattr(GenerationPipeline, "_process_single_mask", fake_process)
    monkeypatch.setattr(
        "clothes_changer.ml.pipeline.get_inpaint_engine",
        lambda: type("Engine", (), {"load": lambda *a, **k: None})(),
    )

    result, _, _ = GenerationPipeline().generate(
        source,
        person_mask=person,
        clothes_mask=clothes,
        prompt="prompt",
        negative_prompt="negative",
        username="test",
    )

    assert result.size == source.size
