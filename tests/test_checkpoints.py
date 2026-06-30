from pathlib import Path

import torch
from safetensors.torch import save_file

from outfit_studio.ml.checkpoints import (
    checkpoint_architecture,
    inpaint_checkpoint_valid,
    is_hub_model_id,
    is_sdxl_checkpoint,
    is_sdxl_model_name,
)


def test_is_hub_model_id():
    assert is_hub_model_id("runwayml/stable-diffusion-inpainting")
    assert not is_hub_model_id("outfit_inpaint_v1.safetensors")


def test_is_sdxl_detection_by_name():
    assert is_sdxl_model_name("photoXL_inpainting_v1.safetensors")
    assert not is_sdxl_model_name("cyberrealistic_v80Inpainting.safetensors")


def test_inpaint_checkpoint_valid_reads_safetensors(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    save_file({"layer.weight": torch.zeros(2, 2)}, path)
    assert inpaint_checkpoint_valid(path)


def test_inpaint_checkpoint_rejects_truncated_file(tmp_path: Path):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"not-a-safetensors-file")
    assert not inpaint_checkpoint_valid(path)


def test_sdxl_architecture_from_tensor_keys(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    save_file({"conditioner.embedders.0.weight": torch.zeros(1)}, path)
    assert is_sdxl_checkpoint("custom.safetensors", path)
    assert checkpoint_architecture("custom.safetensors", path) == "sdxl"
