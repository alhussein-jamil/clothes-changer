"""Tests for Generate run prep: mask bboxes, pose-before-inpaint, prompt embeds."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from outfit_studio.ml.pipeline import GenerationPipeline, _bboxes_from_masks


def test_bboxes_from_masks_finds_connected_components():
    person = np.zeros((100, 200), dtype=np.uint8)
    clothes = np.zeros((100, 200), dtype=np.uint8)
    person[10:40, 10:50] = 1
    clothes[10:40, 10:50] = 1
    person[10:40, 120:180] = 1
    clothes[20:35, 130:170] = 1
    boxes = _bboxes_from_masks(person, clothes)
    assert len(boxes) == 2
    assert boxes[0, 0] < boxes[1, 0]


def test_bboxes_from_masks_empty():
    person = np.zeros((32, 32), dtype=np.uint8)
    clothes = np.zeros((32, 32), dtype=np.uint8)
    boxes = _bboxes_from_masks(person, clothes)
    assert len(boxes) == 0


def test_generate_builds_pose_before_inpaint_load(monkeypatch):
    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "false")
    get_settings.cache_clear()

    source = Image.new("RGB", (128, 128), color=(20, 20, 20))
    person = np.zeros((128, 128), dtype=np.uint8)
    clothes = np.zeros((128, 128), dtype=np.uint8)
    clothes[32:96, 32:96] = 1
    person[24:104, 24:104] = 1

    order: list[str] = []

    class DummyPose:
        device = "cpu"
        sessions_loaded = True

        def get_bboxes(self, image, *, prepare=True):
            order.append("yolo")
            return np.array([[0, 0, image.width, image.height]], dtype=np.float32)

        def estimate_keypoints(self, image, bboxes=None, *, prepare=True):
            order.append("keypoints")
            assert prepare is False
            return np.zeros((1, 134, 2), dtype=np.float32), np.zeros((1, 134), dtype=np.float32)

        def render_skeleton(self, size, keypoints, scores):
            order.append("skeleton")
            return Image.new("RGB", size, color=(0, 255, 0))

        def unload(self):
            order.append("pose_unload")

    class DummyEngine:
        def load(self, *a, **k):
            order.append("inpaint_load")

        def encode_prompt_embeds(self, *a, **k):
            order.append("encode_prompt")
            return {"prompt_embeds": torch.zeros(1, 2, 4)}

        def default_model_id(self):
            return "dummy"

        def model_architecture(self, _model):
            return "sd15"

        def inference_size(self):
            return 64

        def inpaint(self, *a, **kwargs):
            order.append("inpaint")
            assert "prompt_embeds" in kwargs or kwargs.get("prompt_embeds") is not None
            # Called via engine.inpaint(..., prompt_embeds=...)
            return Image.new("RGB", (64, 64), color=(30, 30, 30))

    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.prepare_for_pose", lambda: order.append("prep_pose")
    )
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.prepare_for_inpaint",
        lambda: order.append("prep_inpaint"),
    )
    monkeypatch.setattr("outfit_studio.ml.pipeline.prepare_next_generate", lambda **k: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.ensure_pose_on_gpu", lambda: DummyPose())
    monkeypatch.setattr("outfit_studio.ml.pipeline.get_pose_estimator", lambda: DummyPose())
    monkeypatch.setattr("outfit_studio.ml.pipeline.get_inpaint_engine", lambda: DummyEngine())
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.blend_images_with_enhancements",
        lambda *a, **k: a[0],
    )
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.remove_reflection_padding",
        lambda img, _pad: img,
    )

    GenerationPipeline().generate(
        source,
        person_mask=person,
        clothes_mask=clothes,
        prompt="prompt",
        negative_prompt="negative",
        username="test",
        use_controlnet=True,
    )

    # Pose guide before inpaint load; no mid-run YOLO when editor masks exist.
    assert "yolo" not in order
    assert order.index("keypoints") < order.index("prep_inpaint")
    assert order.index("skeleton") < order.index("prep_inpaint")
    assert order.index("prep_inpaint") < order.index("inpaint_load")
    assert order.index("encode_prompt") < order.index("inpaint")


def test_generate_skips_pose_stack_when_controlnet_off(monkeypatch):
    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "false")
    get_settings.cache_clear()

    source = Image.new("RGB", (64, 64), color=(20, 20, 20))
    person = np.ones((64, 64), dtype=np.uint8)
    clothes = np.ones((64, 64), dtype=np.uint8)

    class BoomPose:
        def get_bboxes(self, *a, **k):
            raise AssertionError("YOLO should not run when ControlNet off + masks exist")

    calls: list[str] = []

    def fake_process(self, *args, **kwargs):
        calls.append("process")
        return Image.new("RGB", (32, 32), color=(30, 30, 30)), {
            "left": 0,
            "top": 0,
            "right": 32,
            "bottom": 32,
        }

    monkeypatch.setattr("outfit_studio.ml.pipeline.prepare_for_pose", lambda: calls.append("pose"))
    monkeypatch.setattr("outfit_studio.ml.pipeline.prepare_for_inpaint", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.prepare_next_generate", lambda **k: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.pipeline.get_pose_estimator", lambda: BoomPose())
    monkeypatch.setattr(GenerationPipeline, "_process_single_mask", fake_process)
    monkeypatch.setattr(
        "outfit_studio.ml.pipeline.get_inpaint_engine",
        lambda: type(
            "Engine",
            (),
            {
                "load": lambda *a, **k: None,
                "encode_prompt_embeds": lambda *a, **k: None,
            },
        )(),
    )

    GenerationPipeline().generate(
        source,
        person_mask=person,
        clothes_mask=clothes,
        prompt="p",
        negative_prompt="n",
        username="test",
        use_controlnet=False,
    )
    assert "pose" not in calls
    assert calls == ["process"]


def test_inpaint_prefers_prompt_embeds():
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._pipe = MagicMock(return_value=MagicMock(images=[Image.new("RGB", (8, 8))]))
    eng._architecture = "sd15"
    eng._warmup_active = False
    eng._use_controlnet = False
    eng._offload_mode = None
    eng._oom_retried = False
    eng.device = torch.device("cpu")
    eng.dtype = torch.float32
    eng.settings = MagicMock()
    eng.settings.content.steps = 1
    eng.settings.content.guidance_scale = 1.0
    eng.settings.content.inference_size = 8
    eng._truncate_prompts = lambda p, n: (p, n)
    eng.inference_size = lambda: 8

    embeds = {
        "prompt_embeds": torch.zeros(1, 2, 4),
        "negative_prompt_embeds": torch.zeros(1, 2, 4),
    }
    eng.inpaint(
        Image.new("RGB", (8, 8)),
        Image.new("L", (8, 8)),
        prompt="hello",
        negative_prompt="bad",
        prompt_embeds=embeds,
    )
    kwargs = eng._pipe.call_args.kwargs
    assert "prompt_embeds" in kwargs
    assert "prompt" not in kwargs
