import logging

import pytest

from clothes_changer.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_log_level_from_env(monkeypatch):
    monkeypatch.setenv("CLOTHES_CHANGER_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("CLOTHES_CHANGER_DEBUG", raising=False)
    settings = get_settings()
    assert settings.log_level == "WARNING"
    assert settings.resolved_log_level() == logging.WARNING


def test_log_level_case_insensitive(monkeypatch):
    monkeypatch.setenv("CLOTHES_CHANGER_LOG_LEVEL", "error")
    settings = get_settings()
    assert settings.log_level == "ERROR"
    assert settings.resolved_log_level() == logging.ERROR


def test_debug_overrides_log_level(monkeypatch):
    monkeypatch.setenv("CLOTHES_CHANGER_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("CLOTHES_CHANGER_DEBUG", "true")
    settings = get_settings()
    assert settings.resolved_log_level() == logging.DEBUG


def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("CLOTHES_CHANGER_LOG_LEVEL", "verbose")
    with pytest.raises(ValueError, match="Invalid log level"):
        get_settings()
