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
os.environ["CLOTHES_CHANGER_OUTPUT_DIR"] = str(Path(_tmp) / "outputs")
os.environ["CLOTHES_CHANGER_REQUIRE_AUTH"] = "false"
os.environ["CLOTHES_CHANGER_MODELS_DIR"] = str(Path(_tmp) / "models")


def _reset_caches() -> None:
    from clothes_changer.config import get_settings

    get_settings.cache_clear()


_reset_caches()


@pytest.fixture
def db():
    from clothes_changer.db.database import Database

    return Database()
