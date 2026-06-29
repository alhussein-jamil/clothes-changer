import io
import logging

from clothes_changer.utils.logging import (
    _encoding_supports,
    log_banner,
    log_duration,
    setup_logging,
    use_unicode_decorations,
)


def test_encoding_supports_ascii_only():
    assert not _encoding_supports("═", "cp1252")
    assert _encoding_supports("hello", "cp1252")


def test_log_banner_on_cp1252_console(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("CLOTHES_CHANGER_ASCII_LOG", raising=False)

    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("clothes_changer")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_banner("Clothes Changer", "tagline — with dash", "http://127.0.0.1:7860")

    output = buffer.buffer.getvalue().decode("cp1252", errors="replace")
    assert "Clothes Changer" in output
    assert "+" in output or "|" in output
    assert "═" not in output


def test_log_duration_ascii_fallback(monkeypatch):
    monkeypatch.setenv("CLOTHES_CHANGER_ASCII_LOG", "1")
    assert not use_unicode_decorations()

    setup_logging(level=logging.DEBUG, force=True)
    logger = logging.getLogger("test.duration")

    with log_duration(logger, "test step"):
        pass
