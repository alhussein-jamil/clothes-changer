"""IP extraction, rate limiting, and security middleware."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

if TYPE_CHECKING:
    from outfit_studio.config import Settings

logger = logging.getLogger(__name__)


def client_ip(request: Request, *, trusted_hops: int = 1) -> str | None:
    """Resolve client IP, honoring X-Forwarded-For behind a reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            # Leftmost is the original client; each proxy appends to the right.
            idx = max(0, len(parts) - trusted_hops - 1)
            return parts[idx]
    if request.client and request.client.host:
        host = request.client.host
        if host and host != "unknown":
            return host
    return None


def device_fingerprint(request: Request) -> str | None:
    raw = request.cookies.get("device_fp", "").strip()
    if not raw or len(raw) > 128:
        return None
    return raw


class RateLimiter:
    """In-memory sliding-window rate limiter (single-process deployments)."""

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._events: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._events[key]
            self._events[key] = [t for t in bucket if t > cutoff]
            if len(self._events[key]) >= self.max_attempts:
                return False
            self._events[key].append(now)
            return True

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob: https:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';",
        )
        return response


class AuthRedirectMiddleware(BaseHTTPMiddleware):
    """Send unauthenticated visitors to /login instead of a bare 401."""

    def __init__(self, app, settings: Settings, resolve_username: Callable[[Request], str | None]):
        super().__init__(app)
        self.settings = settings
        self.resolve_username = resolve_username
        self._public_prefixes = (
            "/login",
            "/auth/",
            "/health",
            "/favicon.ico",
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.require_auth:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in self._public_prefixes):
            return await call_next(request)

        if path == "/" and request.method == "GET" and self.resolve_username(request) is None:
            return RedirectResponse(url="/login", status_code=302)

        return await call_next(request)
