"""Cooperative cancellation for long-running UI and ML operations."""

from __future__ import annotations

import threading
from contextvars import ContextVar

_session_hash: ContextVar[str | None] = ContextVar("_session_hash", default=None)
_cancelled: dict[str, bool] = {}
_lock = threading.Lock()


class OperationCancelled(Exception):
    """Raised when the user clicks Stop for the current session."""


def session_hash_from(request: object | None) -> str | None:
    return getattr(request, "session_hash", None) if request is not None else None


def bind_session(session_hash: str | None) -> None:
    _session_hash.set(session_hash)


def bind_request(request: object | None) -> None:
    bind_session(session_hash_from(request))


def begin_operation(session_hash: str | None) -> None:
    """Mark a new operation for *session_hash* and clear any prior stop request."""
    bind_session(session_hash)
    with _lock:
        if session_hash:
            _cancelled[session_hash] = False


def request_stop(session_hash: str | None) -> None:
    if session_hash:
        with _lock:
            _cancelled[session_hash] = True


def check_cancelled() -> None:
    session = _session_hash.get()
    if session and _cancelled.get(session):
        raise OperationCancelled
