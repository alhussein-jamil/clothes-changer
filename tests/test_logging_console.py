import io
import logging

from outfit_studio.utils.logging import (
    _encoding_supports,
    log_banner,
    log_duration,
    setup_logging,
    use_unicode_decorations,
)


def test_encoding_supports_ascii_only():
    assert not _encoding_supports("═", "cp1252")
    assert _encoding_supports("hello", "cp1252")


def test_use_unicode_respects_ascii_env(monkeypatch):
    monkeypatch.setenv("OUTFIT_STUDIO_ASCII_LOG", "1")
    assert not use_unicode_decorations()


def test_log_banner_ascii_fallback(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("OUTFIT_STUDIO_ASCII_LOG", "1")

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("outfit_studio")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_banner("Outfit Studio", "tagline with dash", "http://127.0.0.1:7860")

    output = buffer.getvalue()
    assert "Outfit Studio" in output
    assert "+" in output or "|" in output
    assert "═" not in output


def test_log_banner_on_cp1252_console(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("OUTFIT_STUDIO_ASCII_LOG", "1")

    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("outfit_studio")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_banner("Outfit Studio", "tagline with dash", "http://127.0.0.1:7860")

    output = buffer.buffer.getvalue().decode("cp1252", errors="strict")
    assert "Outfit Studio" in output
    assert "+" in output or "|" in output
    assert "═" not in output


def test_log_duration_ascii_fallback(monkeypatch):
    monkeypatch.setenv("OUTFIT_STUDIO_ASCII_LOG", "1")
    assert not use_unicode_decorations()

    setup_logging(level=logging.DEBUG, force=True)
    logger = logging.getLogger("test.duration")

    with log_duration(logger, "test step"):
        pass
