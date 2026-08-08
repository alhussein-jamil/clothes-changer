from unittest.mock import MagicMock, patch

from outfit_studio.ml import gpu_memory


def test_both_stacks_fit_when_total_vram_is_large(monkeypatch):
    monkeypatch.setattr(gpu_memory.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_memory, "gpu_memory_gb", lambda: (8.0, 16.0))
    monkeypatch.setattr(gpu_memory, "_inpaint_vram_budget_gb", lambda: 6.0)
    assert gpu_memory.both_stacks_fit_on_gpu() is True


def test_vram_is_tight_below_threshold(monkeypatch):
    monkeypatch.setattr(gpu_memory.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_memory, "gpu_total_gb", lambda: 8.0)
    assert gpu_memory.vram_is_tight() is True
    monkeypatch.setattr(gpu_memory, "gpu_total_gb", lambda: 12.0)
    assert gpu_memory.vram_is_tight() is False


def test_enable_low_vram_guards_slices_on_tight_gpu(monkeypatch):
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._cpu_offload = False
    eng._offload_mode = None
    eng._architecture = "sd15"
    pipe = MagicMock()
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.vram_is_tight", lambda: True)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.gpu_total_gb", lambda: 8.0)
    eng._enable_low_vram_guards(pipe, offload=None)
    pipe.enable_attention_slicing.assert_called()
    pipe.enable_vae_slicing.assert_called()
    pipe.enable_vae_tiling.assert_called()
    assert eng._cpu_offload is False


def test_enable_low_vram_guards_skips_vae_tiling_for_sdxl(monkeypatch):
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._cpu_offload = False
    eng._offload_mode = None
    eng._architecture = "sdxl"
    pipe = MagicMock()
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.vram_is_tight", lambda: True)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.gpu_total_gb", lambda: 8.0)
    eng._enable_low_vram_guards(pipe, offload=None)
    pipe.enable_vae_slicing.assert_called()
    pipe.enable_vae_tiling.assert_not_called()


def test_enable_low_vram_guards_prefers_model_offload(monkeypatch):
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._cpu_offload = False
    eng._offload_mode = None
    eng._architecture = "sd15"
    pipe = MagicMock()
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.gpu_total_gb", lambda: 8.0)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.vram_is_tight", lambda: True)
    assert eng._enable_low_vram_guards(pipe, offload="model") is True
    pipe.enable_model_cpu_offload.assert_called_once()
    pipe.enable_sequential_cpu_offload.assert_not_called()
    assert eng._offload_mode == "model"
    assert eng._cpu_offload is True


def test_enable_low_vram_guards_sequential_when_requested(monkeypatch):
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._cpu_offload = False
    eng._offload_mode = None
    eng._architecture = "sdxl"
    pipe = MagicMock()
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.gpu_total_gb", lambda: 8.0)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.vram_is_tight", lambda: True)
    assert eng._enable_low_vram_guards(pipe, offload="sequential") is True
    pipe.enable_sequential_cpu_offload.assert_called_once()
    assert eng._offload_mode == "sequential"


def test_enable_low_vram_guards_falls_back_without_accelerate(monkeypatch):
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._cpu_offload = False
    eng._offload_mode = None
    eng._architecture = "sd15"
    eng.device = MagicMock(type="cuda")
    pipe = MagicMock()
    pipe.enable_model_cpu_offload.side_effect = ImportError("need accelerate")
    pipe.unet = MagicMock()
    pipe.unet.to = MagicMock(return_value=pipe.unet)
    pipe.controlnet = MagicMock()
    pipe.controlnet.to = MagicMock(return_value=pipe.controlnet)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.free_cuda_cache", lambda: None)
    monkeypatch.setattr("outfit_studio.ml.gpu_memory.vram_is_tight", lambda: True)
    eng._enable_low_vram_guards(pipe, offload="model")
    assert eng._cpu_offload is True
    pipe.unet.to.assert_called()


def test_inference_size_bumps_sdxl():
    from outfit_studio.ml.inpainter import InpaintEngine

    eng = InpaintEngine.__new__(InpaintEngine)
    eng._architecture = "sdxl"
    eng._current_model = "lustifyAPEXInpainting.safetensors"
    eng.settings = MagicMock()
    eng.settings.content.inference_size = 512
    assert eng.inference_size() == 1024

    eng._architecture = "sd15"
    eng._current_model = "cyberrealistic_v80Inpainting.safetensors"
    assert eng.inference_size() == 512


def test_prepare_for_segmentation_keeps_inpaint_when_both_fit(monkeypatch):
    monkeypatch.setattr(gpu_memory, "segmentation_uses_cuda", lambda: True)
    monkeypatch.setattr(gpu_memory, "both_stacks_fit_on_gpu", lambda: True)
    monkeypatch.setattr(gpu_memory, "gpu_free_gb", lambda: 8.0)
    engine = MagicMock(is_loaded=MagicMock(return_value=True))
    with (
        patch("outfit_studio.ml.inpainter.get_inpaint_engine", return_value=engine),
        patch.object(gpu_memory, "release_inpaint_gpu") as release,
    ):
        gpu_memory.prepare_for_segmentation()
    release.assert_not_called()


def test_prepare_for_segmentation_unloads_inpaint_when_tight(monkeypatch):
    monkeypatch.setattr(gpu_memory, "segmentation_uses_cuda", lambda: True)
    monkeypatch.setattr(gpu_memory, "both_stacks_fit_on_gpu", lambda: False)
    engine = MagicMock(is_loaded=MagicMock(return_value=True))
    with (
        patch("outfit_studio.ml.inpainter.get_inpaint_engine", return_value=engine),
        patch.object(gpu_memory, "release_inpaint_gpu") as release,
    ):
        gpu_memory.prepare_for_segmentation()
    release.assert_called_once()


def test_prepare_for_inpaint_skips_seg_release_when_on_cpu_and_unloaded(monkeypatch):
    monkeypatch.setattr(gpu_memory, "segmentation_uses_cuda", lambda: False)
    monkeypatch.setattr(gpu_memory, "free_cuda_cache", lambda: None)
    monkeypatch.setattr(gpu_memory, "gpu_memory_gb", lambda: (6.0, 8.0))
    monkeypatch.setattr(gpu_memory, "_inpaint_vram_budget_gb", lambda: 6.0)
    with (
        patch.object(gpu_memory, "release_pose_gpu"),
        patch.object(gpu_memory, "release_segmentation_gpu") as release,
        patch("outfit_studio.ml.segmentor.get_segmentor") as get_seg,
    ):
        get_seg.return_value = MagicMock(is_loaded=MagicMock(return_value=False))
        gpu_memory.prepare_for_inpaint()
    release.assert_not_called()


def test_prepare_for_inpaint_releases_pose(monkeypatch):
    monkeypatch.setattr(gpu_memory, "free_cuda_cache", lambda: None)
    monkeypatch.setattr(gpu_memory, "gpu_memory_gb", lambda: (6.0, 8.0))
    monkeypatch.setattr(gpu_memory, "gpu_free_gb", lambda: 6.0)
    monkeypatch.setattr(gpu_memory, "_inpaint_vram_budget_gb", lambda: 6.0)
    monkeypatch.setattr(gpu_memory, "segmentation_uses_cuda", lambda: False)
    with (
        patch.object(gpu_memory, "release_pose_gpu") as release_pose,
        patch("outfit_studio.ml.segmentor.get_segmentor") as get_seg,
    ):
        get_seg.return_value = MagicMock(is_loaded=MagicMock(return_value=False))
        gpu_memory.prepare_for_inpaint()
    release_pose.assert_called_once()


def test_prepare_for_pose_unloads_inpaint_when_vram_tight(monkeypatch):
    monkeypatch.setattr(gpu_memory.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_memory, "free_cuda_cache", lambda: None)
    state = {"n": 0}

    def free_gb() -> float:
        state["n"] += 1
        return 0.2 if state["n"] <= 2 else 2.0

    monkeypatch.setattr(gpu_memory, "gpu_free_gb", free_gb)
    engine = MagicMock(is_loaded=MagicMock(return_value=True))
    with (
        patch("outfit_studio.ml.inpainter.get_inpaint_engine", return_value=engine),
        patch.object(gpu_memory, "release_inpaint_gpu") as release_inpaint,
        patch.object(gpu_memory, "release_segmentation_gpu") as release_seg,
    ):
        gpu_memory.prepare_for_pose()
    release_inpaint.assert_called_once()
    release_seg.assert_not_called()
