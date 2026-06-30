"""Runtime configuration via environment variables.

Deployment settings (host, paths, auth, debug) live in ``.env``.
Content and ML defaults (prompts, models, generation) live in ``config/content*.yaml``.
"""

import logging
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from outfit_studio import content_config
from outfit_studio.constants import DEFAULT_ADMIN_BOOTSTRAP_CREDITS

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PROJECT_ROOT
_DEFAULT_SESSION_SECRET = "change-me-in-production"
_LOCAL_HOST_MARKERS = ("localhost", "127.0.0.1", "[::1]")


class Settings(BaseSettings):
    """Deployment and runtime settings — not duplicated in YAML."""

    model_config = SettingsConfigDict(
        env_prefix="OUTFIT_STUDIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    log_level: str = "INFO"

    models_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    db_path: Path = Path("database.db")
    examples_dir: Path = Path("examples")
    favicon_path: Path = Path("static/favicon.ico")
    logo_path: Path = Path("static/outfit-studio-logo.png")
    enable_sharing: bool = False
    require_auth: bool = True

    max_image_size: int = 1024

    default_admin: str = "admin"
    default_password: str = "admin"
    default_credits: int = DEFAULT_ADMIN_BOOTSTRAP_CREDITS

    # Auth / security (production)
    session_secret: str = _DEFAULT_SESSION_SECRET
    session_cookie_secure: bool = False
    public_base_url: str = "http://localhost:7860"
    allow_local_signup: bool = True
    allow_bootstrap_admin: bool = True
    enforce_single_account_per_ip: bool = True
    enforce_single_account_per_device: bool = True
    trusted_proxy_hops: int = 1
    forwarded_allow_ips: str = "127.0.0.1"
    login_rate_limit: int = 10
    login_rate_window_s: int = 300
    signup_rate_limit: int = 5
    signup_rate_window_s: int = 3600

    pipeline_debug: bool = False
    pipeline_debug_dir: Path = Path("debug-pipeline")
    torch_compile: bool = True
    torch_compile_cache: bool = True
    torch_compile_cache_dir: Path = Path(".cache/torch_compile")
    inductor_cache_dir: Path | None = None

    @property
    def content(self) -> content_config.ContentSettings:
        """Branded copy and ML defaults from config/content*.yaml."""
        return content_config.get_content_settings()

    @property
    def compile_inpaint_size(self) -> int:
        """Fixed square size for every inpaint pass (keeps torch.compile graphs stable)."""
        from outfit_studio.constants import LATENT_ALIGN, MIN_LATENT_SIDE

        aligned = self.content.inference_size // LATENT_ALIGN * LATENT_ALIGN
        return max(aligned, MIN_LATENT_SIDE)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> str:
        name = str(value or "INFO").upper()
        if name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = f"Invalid log level {value!r}; use DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            raise ValueError(msg)
        return name

    def resolved_log_level(self) -> int:
        """Effective logging level (`debug=True` forces DEBUG)."""
        if self.debug:
            return logging.DEBUG
        return getattr(logging, self.log_level)

    def _resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (_PROJECT_ROOT / path).resolve()

    @property
    def resolved_models_dir(self) -> Path:
        return self._resolve(self.models_dir)

    @property
    def resolved_output_dir(self) -> Path:
        return self._resolve(self.output_dir)

    @property
    def resolved_db_path(self) -> Path:
        return self._resolve(self.db_path)

    @property
    def resolved_examples_dir(self) -> Path:
        return self._resolve(self.examples_dir)

    @property
    def resolved_favicon_path(self) -> Path:
        return self._resolve(self.favicon_path)

    @property
    def resolved_logo_path(self) -> Path:
        return self._resolve(self.logo_path)

    @property
    def resolved_static_dir(self) -> Path:
        return self._resolve(Path("static"))

    @property
    def resolved_pipeline_debug_dir(self) -> Path:
        return self._resolve(self.pipeline_debug_dir)

    @property
    def resolved_torch_compile_cache_dir(self) -> Path:
        return self._resolve(self.torch_compile_cache_dir)

    @property
    def resolved_inductor_cache_dir(self) -> Path:
        path = self.inductor_cache_dir or Path(".cache/torchinductor")
        return self._resolve(path)

    @property
    def resolved_session_secret(self) -> str:
        """Effective session signing key (auto-generated for local dev when unset)."""
        if self.session_secret != _DEFAULT_SESSION_SECRET and len(self.session_secret) >= 32:
            return self.session_secret
        if is_local_development(self):
            return _local_dev_session_secret()
        return self.session_secret

    def ensure_dirs(self) -> None:
        for label, path in (
            ("models", self.resolved_models_dir),
            ("outputs", self.resolved_output_dir),
        ):
            created = not path.is_dir()
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "Directory %s: %s%s",
                label,
                path,
                " (created)" if created else "",
            )
        logger.info(
            "Paths ready — models=%s outputs=%s db=%s",
            self.resolved_models_dir,
            self.resolved_output_dir,
            self.resolved_db_path,
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    logger.debug(
        "Settings loaded (host=%s:%s, log_level=%s, debug=%s, inpaint_model=%s)",
        settings.host,
        settings.port,
        settings.log_level,
        settings.debug,
        settings.content.default_inpaint,
    )
    return settings


_WEAK_PASSWORDS = frozenset({"admin", "password", "changeme", "12345678"})


def is_local_development(settings: Settings) -> bool:
    """True when running on a developer machine (not a production deploy)."""
    if settings.debug:
        return True
    base = settings.public_base_url.lower()
    return any(marker in base for marker in _LOCAL_HOST_MARKERS)


@lru_cache
def _local_dev_session_secret() -> str:
    return secrets.token_urlsafe(48)


def validate_deployment_settings(settings: Settings) -> None:
    """Fail fast on unsafe production configuration when auth is required."""
    if not settings.require_auth:
        return

    if is_local_development(settings):
        if settings.session_secret == _DEFAULT_SESSION_SECRET:
            logger.warning(
                "OUTFIT_STUDIO_SESSION_SECRET is unset — using an ephemeral local secret. "
                "Set a stable secret (32+ chars) before production deploy."
            )
        return

    if settings.session_secret == _DEFAULT_SESSION_SECRET or len(settings.session_secret) < 32:
        msg = (
            "Set OUTFIT_STUDIO_SESSION_SECRET to a random string of at least 32 characters "
            "before deploying with authentication enabled"
        )
        raise RuntimeError(msg)

    if (
        settings.allow_bootstrap_admin
        and settings.default_password in _WEAK_PASSWORDS
        and not settings.debug
    ):
        logger.warning(
            "Default admin password is weak — change OUTFIT_STUDIO_DEFAULT_PASSWORD "
            "or set OUTFIT_STUDIO_ALLOW_BOOTSTRAP_ADMIN=false after first deploy"
        )
