import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from outfit_studio.ml.checkpoints import is_sdxl_checkpoint, is_sdxl_model_name
from outfit_studio.ml.inpainter import InpaintEngine


def test_is_sdxl_detection_by_name():
    assert is_sdxl_checkpoint("photoXL_inpainting_v1.safetensors", Path("x"))
    assert is_sdxl_model_name("lustifyAPEXInpainting.safetensors")
    assert not is_sdxl_checkpoint("cyberrealistic_v80Inpainting.safetensors", Path("x"))


def test_list_local_models_from_env(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    for name in (
        "cyberrealistic_v80Inpainting.safetensors",
        "outfit_inpaint_v1.safetensors",
    ):
        save_file({"unet.weight": torch.zeros(1)}, tmp / name)

    monkeypatch.setenv("OUTFIT_STUDIO_MODELS_DIR", str(tmp))
    from outfit_studio.config import get_settings
    from outfit_studio.content_config import ContentSettings

    test_content = ContentSettings(default_inpaint="outfit_inpaint_v1.safetensors")
    monkeypatch.setattr("outfit_studio.content_config.get_content_settings", lambda: test_content)
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
    # Bare Diffusers check_inputs assert must not be treated as inductor failure.
    assert not InpaintEngine._is_compile_runtime_error(AssertionError())
    assert InpaintEngine._is_compile_runtime_error(
        RuntimeError("Error: accessing tensor output of CUDAGraphs")
    )
    assert InpaintEngine._is_compile_runtime_error(
        RuntimeError("torch._inductor.exc.InductorError: boom")
    )
    assert not InpaintEngine._is_compile_runtime_error(ValueError("bad prompt"))


def test_decompile_pipe_restores_orig_mod():
    class Wrapped:
        _orig_mod = object()

    pipe = type("Pipe", (), {"unet": Wrapped(), "controlnet": Wrapped()})()
    assert InpaintEngine._decompile_pipe(pipe) is True
    assert pipe.unet is Wrapped._orig_mod


def test_unwrap_compiled_modules_restores_tt_unet_backup():
    from outfit_studio.ml.tt_runtime import CompiledModuleAdapter

    engine = InpaintEngine()
    eager = object()
    adapter = CompiledModuleAdapter(
        type("C", (), {"close": lambda self: None})(),
        name="unet",
    )
    pipe = type("Pipe", (), {"unet": adapter, "controlnet": None})()
    engine._tt_unet_eager = eager
    assert engine._unwrap_compiled_modules(pipe) is True
    assert pipe.unet is eager
    assert engine._tt_unet_eager is None


def test_sdxl_warmup_skips_step_clamp(monkeypatch):
    engine = InpaintEngine()
    engine._architecture = "sdxl"
    engine._warmup_active = True
    seen = {}

    class DummyPipe:
        def __call__(self, **kwargs):
            from types import SimpleNamespace

            seen["steps"] = kwargs["num_inference_steps"]
            return SimpleNamespace(images=[kwargs["image"]])

    engine._pipe = DummyPipe()
    engine._current_model = "x.safetensors"
    image = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16))
    engine.inpaint(image, mask, "p", "n", steps=1)
    assert seen["steps"] == 1


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


def test_checkpoint_download_headers_include_civitai_bearer(monkeypatch):
    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_CIVITAI_API_TOKEN", "test-token-abc")
    get_settings.cache_clear()
    engine = InpaintEngine()
    headers = engine._checkpoint_download_headers("https://civitai.red/api/download/models/1464918")
    assert headers["Authorization"] == "Bearer test-token-abc"
    assert "User-Agent" in headers
    # Non-civitai URLs stay unauthenticated
    plain = engine._checkpoint_download_headers("https://example.com/model.safetensors")
    assert "Authorization" not in plain
    get_settings.cache_clear()


def test_download_model_unauthorized_mentions_token_env(monkeypatch, tmp_path):
    import requests

    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_MODELS_DIR", str(tmp_path))
    get_settings.cache_clear()

    engine = InpaintEngine()
    monkeypatch.setattr(
        "outfit_studio.ml.inpainter.get_checkpoint_urls",
        lambda: {"missing.safetensors": "https://civitai.red/api/download/models/1"},
    )

    class FakeResponse:
        status_code = 401
        headers = {}

        def raise_for_status(self):
            raise AssertionError("raise_for_status should not run after 401 helper")

        def iter_content(self, *_args, **_kwargs):
            return iter(())

    monkeypatch.setattr(
        "outfit_studio.ml.inpainter.requests.get",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    try:
        engine.download_model(tmp_path / "missing.safetensors")
        raise AssertionError("expected HTTPError")
    except requests.HTTPError as exc:
        assert "OUTFIT_STUDIO_CIVITAI_API_TOKEN" in str(exc)
    finally:
        get_settings.cache_clear()


def test_tensor_torrent_miss_restores_peers_to_cuda(monkeypatch):
    """TT size-gate miss must not leave text_encoder on CPU (breaks encode_prompt)."""
    from unittest.mock import MagicMock

    eng = InpaintEngine.__new__(InpaintEngine)
    eng.device = torch.device("cuda")
    eng.dtype = torch.float16
    eng._tt_unet_eager = None
    eng.settings = MagicMock(
        tensor_torrent_unet=True,
        tensor_torrent_min_params_gb=4.0,
        resolved_tensor_torrent_cache_dir=MagicMock(),
    )
    eng.inference_size = lambda: 512

    class FakeMod(torch.nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.device_name = "cpu"

        def to(self, device, *args, **kwargs):  # noqa: ANN002
            if isinstance(device, torch.device):
                self.device_name = device.type
            elif isinstance(device, str):
                self.device_name = "cuda" if device.startswith("cuda") else device
            return self

    te = FakeMod("text_encoder")
    vae = FakeMod("vae")
    unet = FakeMod("unet")

    class Pipe:
        def __init__(self):
            self.unet = unet
            self.text_encoder = te
            self.text_encoder_2 = None
            self.vae = vae
            self.controlnet = None

    pipe = Pipe()
    monkeypatch.setattr(
        "outfit_studio.ml.gpu_memory.vram_is_tight",
        lambda: True,
    )
    monkeypatch.setattr(
        "outfit_studio.ml.gpu_memory.free_cuda_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        "outfit_studio.ml.tt_runtime.try_compile_unet",
        lambda *a, **k: None,
    )

    assert eng._apply_tensor_torrent(pipe, model_id="tiny.safetensors") is False
    assert te.device_name == "cuda"
    assert vae.device_name == "cuda"


def test_tensor_torrent_hit_keeps_peers_on_cpu_when_tight(monkeypatch):
    from unittest.mock import MagicMock

    eng = InpaintEngine.__new__(InpaintEngine)
    eng.device = torch.device("cuda")
    eng.dtype = torch.float16
    eng._tt_unet_eager = None
    eng.settings = MagicMock(
        tensor_torrent_unet=True,
        tensor_torrent_min_params_gb=4.0,
        resolved_tensor_torrent_cache_dir=MagicMock(),
    )
    eng.inference_size = lambda: 512

    class FakeMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device_name = "cpu"

        def to(self, device, *args, **kwargs):  # noqa: ANN002
            if isinstance(device, torch.device):
                self.device_name = device.type
            elif isinstance(device, str):
                self.device_name = "cuda" if device.startswith("cuda") else device
            return self

    te = FakeMod()
    unet = FakeMod()
    compiled = FakeMod()

    class Pipe:
        def __init__(self):
            self.unet = unet
            self.text_encoder = te
            self.text_encoder_2 = None
            self.vae = None
            self.controlnet = None

    pipe = Pipe()
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.vram_is_tight", lambda: True)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.free_cuda_cache", lambda: None)
    monkeypatch.setattr(
        "outfit_studio.ml.tt_runtime.try_compile_unet",
        lambda *a, **k: compiled,
    )

    assert eng._apply_tensor_torrent(pipe, model_id="huge.safetensors") is True
    assert te.device_name == "cpu"
    assert pipe.unet is compiled
