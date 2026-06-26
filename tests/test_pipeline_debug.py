import numpy as np
from PIL import Image

from clothes_changer.ml.pipeline_debug import PipelineDebugSession


def test_pipeline_debug_session_writes_artifacts(tmp_path, monkeypatch):
    from clothes_changer.config import get_settings

    monkeypatch.setenv("CLOTHES_CHANGER_PIPELINE_DEBUG", "true")
    monkeypatch.setenv("CLOTHES_CHANGER_PIPELINE_DEBUG_DIR", str(tmp_path / "debug"))
    get_settings.cache_clear()

    session = PipelineDebugSession.create(get_settings(), "tester")
    assert session is not None

    img = Image.new("RGB", (16, 16), color=(40, 40, 40))
    person = np.zeros((16, 16), dtype=np.uint8)
    clothes = np.zeros((16, 16), dtype=np.uint8)
    clothes[4:10, 4:10] = 1

    session.save_image("00_source.png", img)
    session.save_mask("01_person_mask.png", person)
    session.save_overlay("02_overlay.png", img, person, clothes)
    session.metadata["model"] = "test.safetensors"
    session.record("person_1", steps=30, cfg=6.5)
    session.save_meta()

    assert (session.root / "00_source.png").is_file()
    assert (session.root / "run_metadata.json").is_file()
    assert (session.root / "02_overlay.png").is_file()


def test_pipeline_debug_disabled_by_default(monkeypatch):
    from clothes_changer.config import get_settings

    monkeypatch.delenv("CLOTHES_CHANGER_PIPELINE_DEBUG", raising=False)
    get_settings.cache_clear()
    assert PipelineDebugSession.create(get_settings(), "guest") is None
