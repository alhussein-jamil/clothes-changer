import numpy as np
from PIL import Image

from outfit_studio.constants import CLOTHES_SEGFORMER_CATEGORIES, PERSON_SEGFORMER_CATEGORIES
from outfit_studio.utils.image import (
    blend_images_with_enhancements,
    clip_bbox,
    get_bounding_box,
    mask_overlay,
    prepare_instance_masks,
    resize_max,
)


def test_segformer_category_constants():
    assert 11 in PERSON_SEGFORMER_CATEGORIES  # sunglasses/glasses → person, not clothes
    assert 4 in CLOTHES_SEGFORMER_CATEGORIES


def test_resize_max():
    img = Image.new("RGB", (2000, 1000))
    out = resize_max(img, 1024)
    assert max(out.size) == 1024


def test_bounding_box_and_prepare_instances():
    person = np.zeros((50, 50), dtype=np.uint8)
    clothes = np.zeros((50, 50), dtype=np.uint8)
    person[10:30, 10:30] = 1
    clothes[15:25, 15:25] = 1
    bbox = get_bounding_box(person | clothes)
    assert bbox[0] == 10
    instances = prepare_instance_masks(
        person,
        clothes,
        np.array([[0, 0, 50, 50]], dtype=np.float32),
    )
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


def test_prepare_instance_masks_splits_two_people():
    person = np.zeros((100, 100), dtype=np.uint8)
    clothes = np.zeros((100, 100), dtype=np.uint8)
    person[10:60, 5:40] = 1
    person[10:60, 60:95] = 1
    clothes[20:50, 15:30] = 1
    clothes[20:50, 70:85] = 1
    bboxes = np.array(
        [[0, 0, 45, 100], [55, 0, 100, 100]],
        dtype=np.float32,
    )
    instances = prepare_instance_masks(person, clothes, bboxes)
    assert len(instances) == 2
    left_person, left_clothes = instances[0]
    right_person, right_clothes = instances[1]
    assert left_clothes.sum() > 0
    assert right_clothes.sum() > 0
    assert left_clothes[:, 50:].sum() == 0
    assert right_clothes[:, :50].sum() == 0
    assert left_person[:, 50:].sum() == 0
    assert right_person[:, :50].sum() == 0


def test_prepare_instance_masks_splits_touching_with_mega_bbox():
    """YOLO often returns one full-frame box plus a partial box for close pairs."""
    person = np.zeros((80, 120), dtype=np.uint8)
    clothes = np.zeros((80, 120), dtype=np.uint8)
    y, x = np.ogrid[:80, :120]
    left = (x - 35) ** 2 + (y - 40) ** 2 < 18**2
    right = (x - 85) ** 2 + (y - 40) ** 2 < 18**2
    person[left | right] = 1
    clothes[left & (x < 40)] = 1
    clothes[right & (x > 80)] = 1
    bboxes = np.array(
        [[0, 0, 119, 79], [70, 10, 110, 70]],
        dtype=np.float32,
    )
    instances = prepare_instance_masks(person, clothes, bboxes)
    assert len(instances) == 2
    left_person = int(instances[0][0].sum())
    right_person = int(instances[1][0].sum())
    assert left_person > 0
    assert right_person > 0
    assert min(left_person, right_person) / max(left_person, right_person) > 0.2


def test_clip_bbox_when_mask_wider_than_image():
    bbox = (100, 400, 400, 700)
    clipped = clip_bbox(bbox, (576, 384))
    assert clipped[1] < clipped[3]
    assert clipped[0] < clipped[2]


def test_align_masks_resizes():
    from outfit_studio.utils.image import align_masks

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
    from outfit_studio.utils.image import composite_crop_onto

    full = Image.new("RGB", (50, 50), color=(0, 0, 255))
    patch = Image.new("RGBA", (20, 20), color=(255, 0, 0, 0))
    patch.putpixel((10, 10), (255, 0, 0, 255))
    out = composite_crop_onto(full, patch, 5, 5)
    assert out.getpixel((5, 5)) == (0, 0, 255)
    assert out.getpixel((15, 15)) == (255, 0, 0)


def test_generation_skips_instances_without_clothes(monkeypatch):
    from outfit_studio.config import get_settings
    from outfit_studio.ml.pipeline import GenerationPipeline

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "false")
    get_settings.cache_clear()

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
        def estimate_keypoints(image):
            return np.zeros((1, 134, 2), dtype=np.float32), np.zeros((1, 134), dtype=np.float32)

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

    monkeypatch.setattr("outfit_studio.ml.pipeline.prepare_for_inpaint", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.get_pose_estimator", lambda: DummyPose())
    monkeypatch.setattr(GenerationPipeline, "_process_single_mask", fake_process)
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.get_inpaint_engine",
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
    from outfit_studio.config import get_settings
    from outfit_studio.ml.pipeline import GenerationPipeline

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "false")
    get_settings.cache_clear()

    source = Image.new("RGB", (1400, 900), color=(20, 20, 20))
    person = np.zeros((900, 1400), dtype=np.uint8)
    clothes = np.zeros((900, 1400), dtype=np.uint8)
    clothes[200:500, 300:700] = 1

    class DummyPose:
        @staticmethod
        def get_bboxes(image):
            return np.array([[0, 0, image.width, image.height]], dtype=np.float32)

        @staticmethod
        def estimate_keypoints(image):
            return np.zeros((1, 134, 2), dtype=np.float32), np.zeros((1, 134), dtype=np.float32)

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

    monkeypatch.setattr("outfit_studio.ml.pipeline.prepare_for_inpaint", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.get_pose_estimator", lambda: DummyPose())
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.prepare_instance_masks",
        lambda person_mask, clothes_mask, bboxes: [(person_mask, clothes_mask)],
    )
    monkeypatch.setattr(GenerationPipeline, "_process_single_mask", fake_process)
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.get_inpaint_engine",
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
