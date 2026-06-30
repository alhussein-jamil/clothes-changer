"""Auth, signup limits, and session security tests."""

import pytest
from starlette.requests import Request

from outfit_studio.auth.security import RateLimiter, client_ip
from outfit_studio.auth.session import SessionManager
from outfit_studio.config import validate_deployment_settings
from outfit_studio.db.database import Database, DatabaseError


@pytest.fixture
def auth_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTFIT_STUDIO_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("OUTFIT_STUDIO_REQUIRE_AUTH", "true")
    monkeypatch.setenv(
        "OUTFIT_STUDIO_SESSION_SECRET",
        "test-secret-key-at-least-32-characters-long",
    )
    from outfit_studio.config import get_settings

    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def auth_db(auth_settings):
    return Database()


def test_default_signup_credits(auth_db):
    auth_db.register_user("newbie", "password123")
    user = auth_db.get_user("newbie")
    assert user is not None
    assert user.credits == 10


def test_single_account_per_ip(auth_db, monkeypatch):
    monkeypatch.setenv("OUTFIT_STUDIO_ENFORCE_SINGLE_ACCOUNT_PER_IP", "true")
    from outfit_studio.config import get_settings

    get_settings.cache_clear()
    auth_db.register_user("user_a", "password123", signup_ip="203.0.113.10")
    with pytest.raises(DatabaseError, match="network"):
        auth_db.register_user("user_b", "password456", signup_ip="203.0.113.10")
    get_settings.cache_clear()


def test_single_account_per_device(auth_db, monkeypatch):
    monkeypatch.setenv("OUTFIT_STUDIO_ENFORCE_SINGLE_ACCOUNT_PER_DEVICE", "true")
    from outfit_studio.config import get_settings

    get_settings.cache_clear()
    auth_db.register_user("user_a", "password123", device_fingerprint="abc123")
    with pytest.raises(DatabaseError, match="device"):
        auth_db.register_user("user_b", "password456", device_fingerprint="abc123")
    get_settings.cache_clear()


def test_session_roundtrip(auth_settings):
    sessions = SessionManager(auth_settings)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
    }
    request = Request(scope)
    token = sessions.create_token("alice")
    scope["headers"] = [(b"cookie", f"outfit_studio_session={token}".encode())]
    request = Request(scope)
    assert sessions.username_from_request(request) == "alice"


def test_rate_limiter_blocks_after_max():
    limiter = RateLimiter(max_attempts=3, window_seconds=60)
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert not limiter.allow("k")


def test_client_ip_from_forwarded():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-forwarded-for", b"203.0.113.1, 10.0.0.1")],
        "query_string": b"",
        "client": ("10.0.0.1", 12345),
    }
    request = Request(scope)
    assert client_ip(request, trusted_hops=1) == "203.0.113.1"


def test_validate_deployment_rejects_weak_secret_in_production(monkeypatch):
    monkeypatch.setenv("OUTFIT_STUDIO_SESSION_SECRET", "short")
    monkeypatch.setenv("OUTFIT_STUDIO_REQUIRE_AUTH", "true")
    monkeypatch.setenv("OUTFIT_STUDIO_PUBLIC_BASE_URL", "https://studio.example.com")
    monkeypatch.setenv("OUTFIT_STUDIO_DEBUG", "false")
    from outfit_studio.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        validate_deployment_settings(settings)
    get_settings.cache_clear()


def test_validate_deployment_allows_placeholder_on_localhost(monkeypatch):
    monkeypatch.setenv("OUTFIT_STUDIO_SESSION_SECRET", "change-me-in-production")
    monkeypatch.setenv("OUTFIT_STUDIO_REQUIRE_AUTH", "true")
    monkeypatch.setenv("OUTFIT_STUDIO_PUBLIC_BASE_URL", "http://localhost:7860")
    from outfit_studio.config import get_settings, validate_deployment_settings

    get_settings.cache_clear()
    settings = get_settings()
    validate_deployment_settings(settings)
    assert len(settings.resolved_session_secret) >= 32
    get_settings.cache_clear()


def test_password_min_length(auth_db):
    with pytest.raises(DatabaseError, match="at least 8"):
        auth_db.register_user("short", "abc")
