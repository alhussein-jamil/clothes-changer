import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_tmp = tempfile.mkdtemp()
os.environ["OUTFIT_STUDIO_DB_PATH"] = str(Path(_tmp) / "test.db")
os.environ["OUTFIT_STUDIO_OUTPUT_DIR"] = str(Path(_tmp) / "outputs")
os.environ["OUTFIT_STUDIO_REQUIRE_AUTH"] = "false"
os.environ["OUTFIT_STUDIO_MODELS_DIR"] = str(Path(_tmp) / "models")


def _reset_caches() -> None:
    from outfit_studio.config import get_settings
    from outfit_studio.content_config import clear_content_config_cache

    get_settings.cache_clear()
    clear_content_config_cache()


_reset_caches()


@pytest.fixture(autouse=True)
def _isolated_config_caches():
    _reset_caches()
    yield
    _reset_caches()


@pytest.fixture
def db():
    from outfit_studio.db.database import Database

    return Database()
