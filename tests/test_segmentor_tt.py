"""Human parser + TensorTorrent static-shape wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from PIL import Image

from outfit_studio.constants import HUMAN_PARSER_TT_SIZE
from outfit_studio.ml.segmentor import ClothesSegmentor


def test_segmentor_tt_forces_processor_square(monkeypatch):
    """TT artifacts are static-shape — processor must emit the compiled HxW."""
    seg = ClothesSegmentor.__new__(ClothesSegmentor)
    seg.settings = MagicMock()
    seg.settings.content.clothes_confidence = 0.15
    seg.settings.content.min_component_area = 32
    seg.settings.content.clothes_edge_grow_px = 0
    seg.device = "cpu"
    seg._tt_input_size = HUMAN_PARSER_TT_SIZE
    seg._lock = __import__("threading").RLock()

    captured: dict = {}

    def fake_processor(*, images, return_tensors, size=None):
        captured["size"] = size
        batch = MagicMock()
        batch.to.return_value = {
            "pixel_values": torch.zeros(1, 3, HUMAN_PARSER_TT_SIZE, HUMAN_PARSER_TT_SIZE)
        }
        return batch

    model = MagicMock()
    logits = torch.zeros(1, 18, 32, 32)
    logits[0, 3, 8:24, 8:24] = 5.0
    logits[0, 1, 4:28, 4:28] = 5.0
    model.return_value = MagicMock(logits=logits)

    seg._processor = fake_processor
    seg._model = model
    monkeypatch.setattr(seg, "_load", lambda: None)

    img = Image.new("RGB", (800, 1200), color=(120, 90, 70))
    person, clothes = seg.segment(img)

    assert captured["size"] == {
        "height": HUMAN_PARSER_TT_SIZE,
        "width": HUMAN_PARSER_TT_SIZE,
    }
    assert person.shape == (1200, 800)
    assert clothes.shape == (1200, 800)


def test_parser_call_resizes_mismatched_pixels(tmp_path, monkeypatch):
    """Safety net inside TT adapter: wrong H/W is resized before compiled forward."""
    from outfit_studio.ml import tt_runtime

    seen = {}

    class FakeCompiled(torch.nn.Module):
        def forward(self, pixels):
            seen["shape"] = tuple(pixels.shape)
            return torch.zeros(1, 18, 16, 16)

    # Build only the call closure via try_compile path stubs
    image_size = 64

    def fake_compile_or_load(
        module,
        *,
        example_inputs,
        artifact_dir,
        name,
        config=None,
        call=None,
        passthrough_attrs=None,
        strict=False,
    ):
        return tt_runtime.CompiledModuleAdapter(
            FakeCompiled(),
            name=name,
            config=config,
            call=call,
            passthrough_attrs=passthrough_attrs,
        )

    monkeypatch.setattr(tt_runtime, "compile_or_load_module", fake_compile_or_load)
    monkeypatch.setattr(tt_runtime, "tensor_torrent_available", lambda: True)

    class Tiny(torch.nn.Module):
        def forward(self, pixel_values=None):
            return type("O", (), {"logits": torch.zeros(1, 18, 8, 8)})()

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

    adapter = tt_runtime.try_compile_human_parser(
        Tiny(),
        model_id="x",
        cache_root=tmp_path,
        image_size=image_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
        min_params_gb=0.0,
    )
    assert adapter is not None
    out = adapter(pixel_values=torch.randn(1, 3, 96, 48))
    assert seen["shape"] == (1, 3, image_size, image_size)
    assert out.logits.shape[1] == 18
