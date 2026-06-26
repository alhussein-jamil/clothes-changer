import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from clothes_changer.ml.checkpoints import is_sdxl_checkpoint
from clothes_changer.ml.inpainter import InpaintEngine


def test_is_sdxl_detection_by_name():
    assert is_sdxl_checkpoint("lustifySDXLNSFW_apexINPAINTING.safetensors", Path("x"))
    assert not is_sdxl_checkpoint("cyberrealistic_v80Inpainting.safetensors", Path("x"))


def test_list_local_models_from_env(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "cyberrealistic_v80Inpainting.safetensors").write_bytes(b"x" * 100)
    (tmp / "realisticVisionV60B1_v51HyperInpaintVAE.safetensors").write_bytes(b"x" * 100)

    monkeypatch.setenv("CLOTHES_CHANGER_MODELS_DIR", str(tmp))
    monkeypatch.setenv(
        "CLOTHES_CHANGER_INPAINT_MODEL", "realisticVisionV60B1_v51HyperInpaintVAE.safetensors"
    )
    from clothes_changer.config import get_settings

    get_settings.cache_clear()

    engine = InpaintEngine()
    models = engine.list_models()
    local_models = [m for m in models if m["source"] == "local"]
    assert len(local_models) == 2
    assert engine.default_model_id() == "realisticVisionV60B1_v51HyperInpaintVAE.safetensors"


def test_inpaint_keeps_existing_loaded_pipeline(monkeypatch):
    engine = InpaintEngine()
    image = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16))

    class DummyPipe:
        def __call__(self, **kwargs):
            return SimpleNamespace(images=[kwargs["image"]])

    engine._pipe = DummyPipe()
    engine._current_model = "selected-model.safetensors"

    def fail_load(*args, **kwargs):
        raise AssertionError("inpaint() should not reload the default model")

    monkeypatch.setattr(engine, "load", fail_load)
    result = engine.inpaint(image, mask, "prompt", "negative")

    assert result.size == image.size
    assert engine._current_model == "selected-model.safetensors"
