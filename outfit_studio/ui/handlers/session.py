"""Session and auth label handlers for GradioApp."""

from __future__ import annotations

import gradio as gr


class SessionHandlersMixin:
    def _session_username(self, request: gr.Request | None = None) -> str | None:
        """Logged-in user, or default admin when auth is disabled."""
        name = getattr(request, "username", None) if request is not None else None
        if name:
            return name
        if not self.settings.require_auth:
            return self.settings.default_admin
        return None

    def _user_label(self, request: gr.Request) -> str:
        name = self._session_username(request) or "Guest"
        return f"Welcome, {name}"

    def _credits_label(self, request: gr.Request) -> str:
        username = self._session_username(request)
        if not username:
            return "0 credits"
        user = self.db.get_user(username)
        if not user:
            return "0 credits"
        if user.is_admin:
            return "Unlimited credits (admin)"
        return f"{user.credits} credits"

    def _effective_debug_dir(
        self, request: gr.Request | None, debug_session_dir: str | None
    ) -> str | None:
        if not self.is_admin(request):
            return None
        return debug_session_dir

    def is_admin(self, request: gr.Request | None = None) -> bool:
        username = self._session_username(request)
        if not username:
            return False
        user = self.db.get_user(username)
        return bool(user and user.is_admin)
