import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_tmp = tempfile.mkdtemp()
os.environ["CLOTHES_CHANGER_DB_PATH"] = str(Path(_tmp) / "test.db")
os.environ["CLOTHES_CHANGER_SECRET_KEY"] = "test-secret"
os.environ["CLOTHES_CHANGER_OUTPUT_DIR"] = str(Path(_tmp) / "outputs")

_DEFAULT_CHECKPOINT = "realisticVisionV60B1_v51HyperInpaintVAE.safetensors"
_project_models = (PROJECT_ROOT / "models").resolve()


def _resolve_models_dir() -> Path:
    """Prefer project models/ when the default checkpoint exists; else download or use tmp."""
    ckpt = _project_models / _DEFAULT_CHECKPOINT
    if ckpt.is_file():
        return _project_models
    try:
        from clothes_changer.scripts.download_models import (
            download_cloth_segm,
            download_default_inpaint_checkpoint,
        )

        _project_models.mkdir(parents=True, exist_ok=True)
        download_cloth_segm(_project_models)
        download_default_inpaint_checkpoint(_project_models)
        if ckpt.is_file():
            return _project_models
    except Exception:
        pass
    return Path(_tmp) / "models"


os.environ["CLOTHES_CHANGER_MODELS_DIR"] = str(_resolve_models_dir())


def _reset_caches() -> None:
    from clothes_changer.config import get_settings

    get_settings.cache_clear()


_reset_caches()


@pytest.fixture
def db():
    from clothes_changer.db.database import Database

    return Database()
