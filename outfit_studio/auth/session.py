"""Signed session cookies for authenticated users."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

if TYPE_CHECKING:
    from starlette.requests import Request

    from outfit_studio.config import Settings

logger = logging.getLogger(__name__)

SESSION_COOKIE = "outfit_studio_session"
DEVICE_COOKIE = "device_fp"
SESSION_MAX_AGE_S = 60 * 60 * 24 * 14  # 14 days


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self._serializer = URLSafeTimedSerializer(
            settings.resolved_session_secret,
            salt="outfit-studio-session",
        )
        self._secure_cookie = settings.session_cookie_secure

    def create_token(self, username: str) -> str:
        return self._serializer.dumps({"username": username})

    def verify_token(self, token: str) -> str | None:
        try:
            data = self._serializer.loads(token, max_age=SESSION_MAX_AGE_S)
        except SignatureExpired:
            logger.info("Session expired")
            return None
        except BadSignature:
            logger.warning("Invalid session signature")
            return None
        username = data.get("username")
        if not isinstance(username, str) or not username.strip():
            return None
        return username.strip()

    def username_from_request(self, request: Request) -> str | None:
        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            return None
        return self.verify_token(token)

    def set_session_cookie(self, response, username: str) -> None:
        response.set_cookie(
            key=SESSION_COOKIE,
            value=self.create_token(username),
            max_age=SESSION_MAX_AGE_S,
            httponly=True,
            secure=self._secure_cookie,
            samesite="lax",
            path="/",
        )

    def clear_session_cookie(self, response) -> None:
        response.delete_cookie(SESSION_COOKIE, path="/")
