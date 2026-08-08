"""TensorTorrent integration unit tests (no full SD compile)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from outfit_studio.ml import tt_runtime


def test_strip_export_device_asserts_removes_nodes():
    class FakeGraph:
        def __init__(self):
            self.nodes = []

        def lint(self):
            return None

        def erase_node(self, node):
            self.nodes.remove(node)

    class FakeGM:
        def __init__(self, graph):
            self.graph = graph

        def recompile(self):
            return None

    class Node:
        def __init__(self, op, target):
            self.op = op
            self.target = target

    g = FakeGraph()
    g.nodes = [
        Node("call_function", "aten._assert_tensor_metadata.default"),
        Node("call_function", "aten.add.Tensor"),
        Node("call_function", "aten._assert_tensor_metadata.default"),
    ]
    exported = SimpleNamespace(graph_module=FakeGM(g))
    removed = tt_runtime.strip_export_device_asserts(exported)
    assert removed == 2
    assert len(g.nodes) == 1
    assert "add" in str(g.nodes[0].target)


def test_artifact_dir_for_is_stable(tmp_path: Path):
    a = tt_runtime.artifact_dir_for(
        tmp_path, component="unet", model_id="cyber.safetensors", shape_key="b2_c9_s64"
    )
    b = tt_runtime.artifact_dir_for(
        tmp_path, component="unet", model_id="cyber.safetensors", shape_key="b2_c9_s64"
    )
    c = tt_runtime.artifact_dir_for(
        tmp_path, component="unet", model_id="other.safetensors", shape_key="b2_c9_s64"
    )
    assert a == b
    assert a != c
    assert a.parent.name == "unet"


def test_latency_compile_config_is_fast_default():
    cfg = tt_runtime.latency_compile_config(cache_dir=Path("/tmp/tt-cache"))
    assert cfg.objective.value == "latency"
    assert cfg.allow_nvme_streaming is False
    assert cfg.use_torch_compile is True
    assert cfg.prefer_direct_path is True
    assert cfg.measure_regions is False
    if torch.cuda.is_available():
        assert cfg.allow_cpu is False
        assert cfg.allow_gpu is True
    else:
        assert cfg.allow_cpu is True
        assert cfg.allow_gpu is False


def test_streaming_compile_config_enables_nvme():
    cfg = tt_runtime.streaming_compile_config(cache_dir=Path("/tmp/tt-cache"))
    assert cfg.objective.value == "memory"
    assert cfg.allow_nvme_streaming is True
    assert cfg.use_torch_compile is False


def test_memory_compile_config_aliases_latency():
    a = tt_runtime.memory_compile_config(cache_dir=Path("/tmp/tt-cache"))
    b = tt_runtime.latency_compile_config(cache_dir=Path("/tmp/tt-cache"))
    assert a.objective == b.objective
    assert a.allow_nvme_streaming == b.allow_nvme_streaming


def test_compile_tiny_module_roundtrip(tmp_path: Path):
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 8)
    artifact_dir = tmp_path / "tiny"
    compiled = tt_runtime.compile_or_load_module(
        model,
        example_inputs=(x,),
        artifact_dir=artifact_dir,
        name="tiny",
    )
    assert tt_runtime.artifact_ready(artifact_dir)
    with torch.inference_mode():
        out = compiled(x)
    assert out.shape == (2, 4)
    # Reload from cache
    reloaded = tt_runtime.compile_or_load_module(
        model,
        example_inputs=(x,),
        artifact_dir=artifact_dir,
        name="tiny",
    )
    with torch.inference_mode():
        out2 = reloaded(x)
    assert out2.shape == out.shape
    compiled.close()
    reloaded.close()


def test_try_compile_or_load_returns_none_on_failure(tmp_path: Path, monkeypatch):
    class Boom(nn.Module):
        def forward(self, x):
            return x

    def raise_compile(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(tt_runtime, "compile_or_load_module", raise_compile)
    result = tt_runtime.try_compile_or_load(
        Boom(),
        example_inputs=(torch.randn(1, 4),),
        artifact_dir=tmp_path / "x",
        name="boom",
    )
    assert result is None


def test_should_use_tensor_torrent_size_gate():
    small = nn.Linear(8, 8)
    assert tt_runtime.should_use_tensor_torrent(small, min_params_gb=4.0, component="t") is False
    assert tt_runtime.should_use_tensor_torrent(small, min_params_gb=0.0, component="t") is True


def test_try_compile_unet_skips_below_size_gate(tmp_path):
    class Cfg:
        in_channels = 9
        cross_attention_dim = 768
        addition_embed_type = None

    class Fake(torch.nn.Module):
        config = Cfg()

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):
            return args[0]

    out = tt_runtime.try_compile_unet(
        Fake(),
        model_id="x.safetensors",
        cache_root=tmp_path,
        latent_side=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        enabled=True,
        min_params_gb=4.0,
    )
    assert out is None


def test_compile_or_load_passes_strict_false(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class Tiny(nn.Module):
        def forward(self, x):
            return x

    class FakeCompiled(nn.Module):
        def forward(self, x):
            return x

        def close(self):
            return None

    def fake_capture(module, example_inputs, *, strict=True):
        captured["strict"] = strict
        return object()

    def fake_compile_exported(*_a, **_k):
        return FakeCompiled()

    monkeypatch.setattr("tensortorrent.capture_module", fake_capture)
    monkeypatch.setattr("tensortorrent.compile_exported", fake_compile_exported)
    monkeypatch.setattr(
        "tensortorrent.load_compiled",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    x = torch.randn(2, 4)
    out = tt_runtime.compile_or_load_module(
        Tiny(),
        example_inputs=(x,),
        artifact_dir=tmp_path / "strict",
        name="tiny",
        strict=False,
    )
    assert captured["strict"] is False
    assert out(x).shape == (2, 4)
    out.close()


def test_unet_is_sdxl_detection():
    class Cfg:
        addition_embed_type = "text_time"
        cross_attention_dim = 2048

    class Fake:
        config = Cfg()

    assert tt_runtime.unet_is_sdxl(Fake()) is True

    class Cfg15:
        addition_embed_type = None
        cross_attention_dim = 768

    class Fake15:
        config = Cfg15()

    assert tt_runtime.unet_is_sdxl(Fake15()) is False


def test_unet_example_inputs_use_batched_timestep():
    class Cfg:
        in_channels = 9
        cross_attention_dim = 768
        addition_embed_type = None

    class Fake:
        config = Cfg()

    sample, timestep, enc = tt_runtime.unet_example_inputs(
        Fake(),
        latent_side=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert sample.shape == (2, 9, 64, 64)
    assert timestep.shape == (2,)
    assert enc.shape == (2, 77, 768)


def test_unet_example_inputs_sdxl_includes_added_cond():
    class Cfg:
        in_channels = 4
        cross_attention_dim = 2048
        addition_embed_type = "text_time"
        addition_time_embed_dim = 256
        projection_class_embeddings_input_dim = 2816

    class Fake:
        config = Cfg()

    sample, timestep, enc, text_embeds, time_ids = tt_runtime.unet_example_inputs(
        Fake(),
        latent_side=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert sample.shape == (2, 4, 128, 128)
    assert enc.shape == (2, 77, 2048)
    assert text_embeds.shape == (2, 1280)
    assert time_ids.shape == (2, 6)


def test_prepare_unet_timestep_expands_scalar_on_sample_device():
    sample = torch.randn(2, 9, 8, 8)
    # Diffusers often keeps scheduler timesteps on CPU
    t = tt_runtime.prepare_unet_timestep(sample, torch.tensor(500))
    assert t.shape == (2,)
    assert t.device == sample.device
    assert int(t[0]) == 500


def test_unet_diffusers_call_prepares_batched_timestep():
    seen = {}

    class FakeCompiled(torch.nn.Module):
        def forward(self, sample, timestep, enc):
            seen["t_shape"] = tuple(timestep.shape)
            seen["t_device"] = str(timestep.device)
            return torch.zeros_like(sample[:, :4])

    adapter = tt_runtime.CompiledModuleAdapter(
        FakeCompiled(),
        name="unet",
        call=tt_runtime._unet_diffusers_call,
    )
    sample = torch.randn(2, 9, 8, 8)
    enc = torch.randn(2, 77, 768)
    # Scalar / CPU timestep like Diffusers scheduler
    out = adapter(sample, torch.tensor(999), encoder_hidden_states=enc, return_dict=False)
    assert seen["t_shape"] == (2,)
    assert seen["t_device"] == str(sample.device)
    assert isinstance(out, tuple)


def test_unet_diffusers_call_passes_sdxl_added_cond():
    seen = {}

    class FakeCompiled(torch.nn.Module):
        def forward(self, sample, timestep, enc, text_embeds, time_ids):
            seen["te"] = tuple(text_embeds.shape)
            seen["tid"] = tuple(time_ids.shape)
            return torch.zeros_like(sample[:, :4])

    adapter = tt_runtime.CompiledModuleAdapter(
        FakeCompiled(),
        name="unet",
        call=lambda c, *a, **k: tt_runtime._unet_diffusers_call(c, *a, expects_sdxl=True, **k),
    )
    sample = torch.randn(2, 4, 8, 8)
    enc = torch.randn(2, 77, 2048)
    text_embeds = torch.randn(2, 1280)
    time_ids = torch.zeros(2, 6)
    out = adapter(
        sample,
        torch.tensor(999),
        encoder_hidden_states=enc,
        added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        return_dict=False,
    )
    assert seen["te"] == (2, 1280)
    assert seen["tid"] == (2, 6)
    assert isinstance(out, tuple)


def test_unet_diffusers_call_sdxl_requires_added_cond():
    adapter = tt_runtime.CompiledModuleAdapter(
        nn.Identity(),
        name="unet",
        call=lambda c, *a, **k: tt_runtime._unet_diffusers_call(c, *a, expects_sdxl=True, **k),
    )
    sample = torch.randn(2, 4, 8, 8)
    enc = torch.randn(2, 77, 2048)
    try:
        adapter(sample, torch.tensor(999), encoder_hidden_states=enc, return_dict=False)
        raise AssertionError("expected TypeError")
    except TypeError as exc:
        assert "text_embeds" in str(exc)


def test_artifact_ready_requires_exported_pt2(tmp_path: Path):
    d = tmp_path / "art"
    d.mkdir()
    (d / "compile_config.json").write_text("{}")
    (d / "portable.json").write_text("{}")
    assert tt_runtime.artifact_ready(d) is False
    (d / "exported.pt2").write_bytes(b"x")
    assert tt_runtime.artifact_ready(d) is True


def test_try_compile_unet_sdxl_not_hard_skipped(tmp_path, monkeypatch):
    class Cfg:
        in_channels = 4
        cross_attention_dim = 2048
        addition_embed_type = "text_time"
        addition_time_embed_dim = 256
        projection_class_embeddings_input_dim = 2816

    class Fake(torch.nn.Module):
        config = Cfg()

        def __init__(self):
            super().__init__()
            # Force size gate open without 4 GiB of params.
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):
            return (args[0][:, :4],)

    captured: dict[str, object] = {}

    def fake_try_compile_or_load(module, *, example_inputs, **kwargs):
        captured["n_inputs"] = len(example_inputs)
        captured["shapes"] = [tuple(t.shape) for t in example_inputs]
        return tt_runtime.CompiledModuleAdapter(module, name="unet")

    monkeypatch.setattr(tt_runtime, "try_compile_or_load", fake_try_compile_or_load)
    monkeypatch.setattr(tt_runtime, "should_use_tensor_torrent", lambda *_a, **_k: True)

    out = tt_runtime.try_compile_unet(
        Fake(),
        model_id="lustify-sdxl.safetensors",
        cache_root=tmp_path,
        latent_side=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
        enabled=True,
        min_params_gb=4.0,
    )
    assert out is not None
    assert captured["n_inputs"] == 5
    assert captured["shapes"][3] == (2, 1280)
    assert captured["shapes"][4] == (2, 6)
