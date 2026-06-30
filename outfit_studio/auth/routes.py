"""Login and registration HTTP routes."""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.responses import Response

from outfit_studio.auth.security import RateLimiter, client_ip, device_fingerprint
from outfit_studio.auth.session import SessionManager
from outfit_studio.constants import DEFAULT_NEW_USER_CREDITS, MIN_PASSWORD_LENGTH
from outfit_studio.db.database import Database, DatabaseError

if TYPE_CHECKING:
    from outfit_studio.config import Settings

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_LOGIN_PAGE_STYLE = (_TEMPLATES_DIR / "login.css").read_text(encoding="utf-8")


def _fingerprint_script() -> str:
    return """
<script>
(function () {
  function hash(s) {
    var h = 0;
    for (var i = 0; i < s.length; i++) h = ((h << 5) - h) + s.charCodeAt(i) | 0;
    return Math.abs(h).toString(16);
  }
  var parts = [
    navigator.userAgent || '',
    navigator.language || '',
    screen.width + 'x' + screen.height,
    screen.colorDepth || ''
  ].join('|');
  document.cookie = 'device_fp=' + hash(parts) + '; path=/; SameSite=Lax; max-age=31536000';
})();
</script>
"""


def _render_page(
    *,
    title: str,
    mode: str,
    error: str | None = None,
    message: str | None = None,
) -> str:
    err = f'<div class="error">{html.escape(error)}</div>' if error else ""
    msg = f'<div class="success">{html.escape(message)}</div>' if message else ""
    signin_active = "active" if mode == "signin" else ""
    signup_active = "active" if mode == "signup" else ""
    if mode == "signin":
        form = """
        <form method="post" action="/auth/login">
          <label for="username">Username</label>
          <input id="username" name="username" required autocomplete="username" />
          <label for="password">Password</label>
          <input id="password" name="password" type="password" required autocomplete="current-password" />
          <button class="primary" type="submit">Sign in</button>
        </form>
        """
    else:
        form = f"""
        <form method="post" action="/auth/register">
          <label for="username">Username</label>
          <input id="username" name="username" required autocomplete="username" minlength="3" maxlength="32" pattern="[A-Za-z0-9_.-]+" />
          <label for="password">Password</label>
          <input id="password" name="password" type="password" required autocomplete="new-password" minlength="{MIN_PASSWORD_LENGTH}" />
          <label for="password2">Confirm password</label>
          <input id="password2" name="password2" type="password" required autocomplete="new-password" minlength="{MIN_PASSWORD_LENGTH}" />
          <button class="primary" type="submit">Create account ({DEFAULT_NEW_USER_CREDITS} credits)</button>
        </form>
        """
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>{html.escape(title)}</title>
<style>{_LOGIN_PAGE_STYLE}</style>
{_fingerprint_script()}
</head><body>
<div class="card">
<h1>{html.escape(title)}</h1>
<p class="sub">Sign in to generate outfits. One account per device and network.</p>
<div class="tabs">
  <a class="{signin_active}" href="/login?mode=signin">Sign in</a>
  <a class="{signup_active}" href="/login?mode=signup">Sign up</a>
</div>
{err}{msg}
{form}
</div>
</body></html>"""


def _login_redirect(
    response: Response, sessions: SessionManager, username: str
) -> RedirectResponse:
    redirect = RedirectResponse(url="/", status_code=303)
    sessions.set_session_cookie(redirect, username)
    return redirect


def register_auth_routes(
    app: FastAPI,
    db: Database,
    settings: Settings,
    sessions: SessionManager,
) -> None:
    login_limiter = RateLimiter(settings.login_rate_limit, settings.login_rate_window_s)
    signup_limiter = RateLimiter(settings.signup_rate_limit, settings.signup_rate_window_s)

    def auth_page(**kwargs: object) -> str:
        return _render_page(**kwargs)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, mode: str = "signin") -> HTMLResponse:
        if sessions.username_from_request(request):
            return RedirectResponse(url="/", status_code=302)  # type: ignore[return-value]
        if mode not in {"signin", "signup"}:
            mode = "signin"
        if mode == "signup" and not settings.allow_local_signup:
            mode = "signin"
        return HTMLResponse(auth_page(title="Outfit Studio", mode=mode))

    @app.post("/auth/login")
    async def auth_login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> RedirectResponse:
        ip = client_ip(request, trusted_hops=settings.trusted_proxy_hops) or "unknown"
        key = f"login:{ip}:{username.strip().lower()}"
        if not login_limiter.allow(key):
            body = auth_page(
                title="Outfit Studio",
                mode="signin",
                error="Too many login attempts. Try again later.",
            )
            return HTMLResponse(body, status_code=429)  # type: ignore[return-value]

        name = username.strip()
        if not db.authenticate(name, password):
            body = auth_page(
                title="Outfit Studio",
                mode="signin",
                error="Invalid username or password.",
            )
            return HTMLResponse(body, status_code=401)  # type: ignore[return-value]

        login_limiter.reset(key)
        db.record_login(name, client_ip(request, trusted_hops=settings.trusted_proxy_hops))
        logger.info("User %r signed in", name)
        return _login_redirect(RedirectResponse(url="/", status_code=303), sessions, name)

    @app.post("/auth/register")
    async def auth_register(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        password2: str = Form(...),
    ) -> Response:
        if not settings.allow_local_signup:
            raise HTTPException(status_code=403, detail="Registration is disabled")

        ip = client_ip(request, trusted_hops=settings.trusted_proxy_hops)
        fp = device_fingerprint(request)
        rate_key = f"signup:{ip or 'unknown'}"
        if not signup_limiter.allow(rate_key):
            body = auth_page(
                title="Outfit Studio",
                mode="signup",
                error="Too many sign-up attempts. Try again later.",
            )
            return HTMLResponse(body, status_code=429)

        name = username.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", name):
            body = auth_page(
                title="Outfit Studio",
                mode="signup",
                error="Username must be 3–32 characters (letters, numbers, _ . -).",
            )
            return HTMLResponse(body, status_code=400)
        if password != password2:
            body = auth_page(
                title="Outfit Studio",
                mode="signup",
                error="Passwords do not match.",
            )
            return HTMLResponse(body, status_code=400)

        try:
            db.register_user(
                name,
                password,
                credits=DEFAULT_NEW_USER_CREDITS,
                signup_ip=ip,
                device_fingerprint=fp,
            )
        except DatabaseError as exc:
            body = auth_page(
                title="Outfit Studio",
                mode="signup",
                error=str(exc),
            )
            return HTMLResponse(body, status_code=400)

        logger.info("Registered user %r (ip=%s)", name, ip)
        return _login_redirect(RedirectResponse(url="/", status_code=303), sessions, name)

    @app.get("/auth/logout")
    async def auth_logout() -> RedirectResponse:
        response = RedirectResponse(url="/login", status_code=303)
        sessions.clear_session_cookie(response)
        return response
