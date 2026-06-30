from unittest.mock import MagicMock, patch

from outfit_studio.ml import gpu_memory


def test_both_stacks_fit_when_total_vram_is_large(monkeypatch):
    monkeypatch.setattr(gpu_memory, "gpu_memory_gb", lambda: (8.0, 16.0))
    monkeypatch.setattr(gpu_memory, "_inpaint_vram_budget_gb", lambda: 6.0)
    assert gpu_memory.both_stacks_fit_on_gpu() is True


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


def test_prepare_for_inpaint_skips_when_segmentation_on_cpu(monkeypatch):
    monkeypatch.setattr(gpu_memory, "segmentation_uses_cuda", lambda: False)
    with patch.object(gpu_memory, "release_segmentation_gpu") as release:
        gpu_memory.prepare_for_inpaint()
    release.assert_not_called()
