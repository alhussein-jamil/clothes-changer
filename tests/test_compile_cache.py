from pathlib import Path

from outfit_studio.ml.compile_cache import artifact_path, cache_key


def test_cache_key_is_stable_for_same_inputs():
    a = cache_key("model.safetensors", "sd15", True)
    b = cache_key("model.safetensors", "sd15", True)
    assert a == b
    assert cache_key("model.safetensors", "sd15", False) != a


def test_artifact_path_uses_ptc_extension(tmp_path: Path):
    path = artifact_path(tmp_path, "runwayml/stable-diffusion-inpainting", "sd15", False)
    assert path.parent == tmp_path
    assert path.suffix == ".ptc"
    assert path.stem.startswith("stable-diffusion-inpainting")
