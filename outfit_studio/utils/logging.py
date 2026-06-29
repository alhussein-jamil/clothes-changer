"""Colorful terminal logging for Outfit Studio."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from outfit_studio.config import get_settings

# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG: ("\033[36m", "DEBUG "),  # cyan
    logging.INFO: ("\033[32m", "INFO  "),  # green
    logging.WARNING: ("\033[33m", "WARN  "),  # yellow
    logging.ERROR: ("\033[31m", "ERROR "),  # red
    logging.CRITICAL: ("\033[35m", "CRIT  "),  # magenta
}

_MODULE_STYLES: dict[str, str] = {
    "outfit_studio.ml": "\033[96m",  # bright cyan
    "outfit_studio.ui": "\033[95m",  # bright magenta
    "outfit_studio.db": "\033[93m",  # bright yellow
    "outfit_studio.utils": "\033[94m",  # bright blue
    "outfit_studio.config": "\033[90m",  # gray
    "outfit_studio.content_config": "\033[90m",
    "outfit_studio.scripts": "\033[92m",  # bright green
}

_BANNER_COLOR = "\033[38;5;45m"  # teal


def supports_color() -> bool:
    """True when stdout is a TTY and NO_COLOR is not set."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _short_module(name: str) -> str:
    if name.startswith("outfit_studio."):
        return name[len("outfit_studio.") :]
    return name


def _module_color(name: str) -> str:
    for prefix, color in _MODULE_STYLES.items():
        if name.startswith(prefix):
            return color
    return "\033[37m"


class ColorFormatter(logging.Formatter):
    """Human-friendly, colorized log lines for development terminals."""

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        level_no = record.levelno
        color, level_label = _LEVEL_STYLES.get(
            level_no, ("\033[37m", record.levelname[:5].ljust(5))
        )
        module = _short_module(record.name)
        message = record.getMessage()

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        if not self.use_color:
            return f"{ts} {level_label.strip():5} {module:28} {message}"

        mod_color = _module_color(record.name)
        return (
            f"{_DIM}{ts}{_RESET} "
            f"{color}{_BOLD}{level_label}{_RESET} "
            f"{mod_color}{module:<28}{_RESET} "
            f"{message}"
        )


def setup_logging(*, level: int | None = None, force: bool = True) -> None:
    """Configure root logger with colorful stdout output."""
    settings = get_settings()
    log_level = level if level is not None else settings.resolved_log_level()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(use_color=supports_color()))

    root = logging.getLogger()
    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)

    root.addHandler(handler)
    root.setLevel(log_level)

    verbose = log_level <= logging.DEBUG
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "diffusers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING if not verbose else logging.INFO)

    logging.getLogger(__name__).debug(
        "Logging initialized (level=%s, color=%s, debug=%s, log_level=%s)",
        logging.getLevelName(log_level),
        supports_color(),
        settings.debug,
        settings.log_level,
    )


def log_banner(title: str, *lines: str) -> None:
    """Print a colorful startup banner."""
    logger = logging.getLogger("outfit_studio")
    width = max(len(title), *(len(line) for line in lines), 44)
    border = "═" * (width + 4)

    if supports_color():
        top = f"{_BANNER_COLOR}╔{border}╗{_RESET}"
        mid_title = (
            f"{_BANNER_COLOR}║{_RESET} {_BOLD}{title:<{width}}{_RESET} {_BANNER_COLOR}║{_RESET}"
        )
        body = [
            f"{_BANNER_COLOR}║{_RESET} {_DIM}{line:<{width}}{_RESET} {_BANNER_COLOR}║{_RESET}"
            for line in lines
        ]
        bottom = f"{_BANNER_COLOR}╚{border}╝{_RESET}"
    else:
        top = f"+{border}+"
        mid_title = f"| {title:<{width}} |"
        body = [f"| {line:<{width}} |" for line in lines]
        bottom = f"+{border}+"

    for line in (top, mid_title, *body, bottom):
        logger.info(line)


@contextmanager
def log_duration(
    logger: logging.Logger,
    label: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> Iterator[None]:
    """Log elapsed wall time for a block (e.g. model load, inference)."""
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    prefix = f"{label} ({extra})" if extra else label
    logger.log(level, "▶ %s …", prefix)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "✓ %s done in %.2fs", prefix, elapsed)
