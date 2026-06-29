"""Application configuration via environment variables."""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from clothes_changer.content_config import get_default_inpaint_model

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PROJECT_ROOT


class Settings(BaseSettings):
    """Central configuration — single source of truth."""

    model_config = SettingsConfigDict(
        env_prefix="CLOTHES_CHANGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = Field(default="dev-secret-change-me")

    models_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    db_path: Path = Path("database.db")
    examples_dir: Path = Path("examples")
    favicon_path: Path = Path("static/favicon.ico")
    enable_sharing: bool = False
    require_auth: bool = False

    max_image_size: int = 1024
    session_hours: int = 8

    default_admin: str = "admin"
    default_password: str = "admin"
    default_credits: int = 100

    # ML
    use_controlnet: bool = True
    controlnet_model: str = "lllyasviel/sd-controlnet-openpose"
    inpaint_model: str = Field(default_factory=get_default_inpaint_model)
    extra_clothes_model: str = "cloth_segm.pth"
    segformer_model: str = "mattmdjaga/segformer_b2_clothes"
    detection_threshold: float = 0.5
    pose_keypoint_threshold: float = 0.3
    pose_mode: str = "balanced"
    inpaint_steps: int = 50
    guidance_scale: float = 6.5
    inference_size: int = 512
    min_inference_size: int = 256

    # Pipeline debug dumps (images per step + run_metadata.json)
    pipeline_debug: bool = False
    pipeline_debug_dir: Path = Path("debug-pipeline")

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
    def resolved_pipeline_debug_dir(self) -> Path:
        return self._resolve(self.pipeline_debug_dir)

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
        settings.inpaint_model,
    )
    return settings
