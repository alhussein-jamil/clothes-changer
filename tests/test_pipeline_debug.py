import json

import numpy as np
from PIL import Image

from outfit_studio.ml.pipeline_debug import PipelineDebugSession


def test_pipeline_debug_session_writes_artifacts(tmp_path, monkeypatch):
    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "true")
    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG_DIR", str(tmp_path / "debug"))
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
    session.record("person_1", inference_steps=30, cfg=6.5)
    session.metadata["inference_steps"] = 30
    session.record("person_2", inference_steps=50)
    session.save_meta()

    meta = json.loads((session.root / "run_metadata.json").read_text(encoding="utf-8"))
    assert isinstance(meta["events"], list)
    assert len(meta["events"]) == 2
    assert meta["inference_steps"] == 30

    assert (session.root / "00_source.png").is_file()
    assert (session.root / "run_metadata.json").is_file()
    assert (session.root / "02_overlay.png").is_file()


def test_open_or_create_unified_user_folder(tmp_path, monkeypatch):
    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "true")
    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG_DIR", str(tmp_path / "debug"))
    get_settings.cache_clear()
    settings = get_settings()

    session, path = PipelineDebugSession.open_or_create(settings, "admin", None)
    assert session is not None
    assert path is not None
    assert session.root.name.startswith("admin_")

    reused, same_path = PipelineDebugSession.open_or_create(settings, "admin", path)
    assert reused is not None
    assert same_path == path
    assert reused.root == session.root

    seg = session.subfolder("segmentation")
    gen = session.subfolder("generation")
    assert seg.root == session.root / "segmentation"
    assert gen.root == session.root / "generation"


def test_pipeline_debug_disabled_by_default(monkeypatch):
    from outfit_studio.config import get_settings

    monkeypatch.setenv("OUTFIT_STUDIO_PIPELINE_DEBUG", "false")
    get_settings.cache_clear()
    assert PipelineDebugSession.create(get_settings(), "guest") is None
