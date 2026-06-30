import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from outfit_studio.ml.checkpoints import is_sdxl_checkpoint
from outfit_studio.ml.inpainter import InpaintEngine


def test_is_sdxl_detection_by_name():
    assert is_sdxl_checkpoint("photoXL_inpainting_v1.safetensors", Path("x"))
    assert not is_sdxl_checkpoint("cyberrealistic_v80Inpainting.safetensors", Path("x"))


def test_list_local_models_from_env(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    for name in (
        "cyberrealistic_v80Inpainting.safetensors",
        "outfit_inpaint_v1.safetensors",
    ):
        save_file({"unet.weight": torch.zeros(1)}, tmp / name)

    monkeypatch.setenv("OUTFIT_STUDIO_MODELS_DIR", str(tmp))
    monkeypatch.setattr(
        "outfit_studio.content_config.get_default_inpaint_model",
        lambda: "outfit_inpaint_v1.safetensors",
    )
    from outfit_studio.config import get_settings
    from outfit_studio.content_config import clear_content_config_cache

    clear_content_config_cache()
    get_settings.cache_clear()

    engine = InpaintEngine()
    models = engine.list_models()
    local_models = [m for m in models if m["source"] == "local"]
    assert len(local_models) == 2
    assert engine.default_model_id() == "outfit_inpaint_v1.safetensors"


def test_is_preparing_false_by_default():
    engine = InpaintEngine()
    assert not engine.is_preparing()


def test_start_background_preload_skips_without_cuda(monkeypatch):
    engine = InpaintEngine()
    monkeypatch.setattr(engine, "device", type("D", (), {"type": "cpu"})())
    engine.start_background_preload()
    assert not engine.is_preparing()
    assert engine._preload_state == "ready"


def test_background_preload_cancellation(monkeypatch):
    engine = InpaintEngine()
    monkeypatch.setattr(engine, "device", type("D", (), {"type": "cuda"})())

    def cancel_during_load(*_args, **_kwargs):
        engine.request_abort()
        engine.checkpoint()

    monkeypatch.setattr(engine, "load", cancel_during_load)
    monkeypatch.setattr(engine, "warmup", lambda: None)
    engine.start_background_preload()
    engine._preload_thread.join(timeout=2)
    assert engine._preload_state == "idle"
    assert not engine.is_preparing()


def test_is_compile_runtime_error():
    assert InpaintEngine._is_compile_runtime_error(AssertionError())
    assert InpaintEngine._is_compile_runtime_error(
        RuntimeError("Error: accessing tensor output of CUDAGraphs")
    )
    assert not InpaintEngine._is_compile_runtime_error(ValueError("bad prompt"))


def test_decompile_pipe_restores_orig_mod():
    class Wrapped:
        _orig_mod = object()

    pipe = type("Pipe", (), {"unet": Wrapped(), "controlnet": Wrapped()})()
    assert InpaintEngine._decompile_pipe(pipe) is True
    assert pipe.unet is Wrapped._orig_mod


def test_inpaint_keeps_existing_loaded_pipeline(monkeypatch):
    engine = InpaintEngine()
    image = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16))

    class DummyPipe:
        def __call__(self, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(images=[kwargs["image"]])

    engine._pipe = DummyPipe()
    engine._current_model = "selected-model.safetensors"

    def fail_load(*args, **kwargs):
        raise AssertionError("inpaint() should not reload the default model")

    monkeypatch.setattr(engine, "load", fail_load)
    result = engine.inpaint(image, mask, "prompt", "negative")

    assert result.size == image.size
    assert engine._current_model == "selected-model.safetensors"
