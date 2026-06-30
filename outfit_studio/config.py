"""Runtime configuration via environment variables.

Deployment settings (host, paths, auth, debug) live in ``.env``.
Content and ML defaults (prompts, models, generation) live in ``config/content*.yaml``.
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from outfit_studio import content_config
from outfit_studio.constants import DEFAULT_ADMIN_BOOTSTRAP_CREDITS

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PROJECT_ROOT


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

    pipeline_debug: bool = False
    pipeline_debug_dir: Path = Path("debug-pipeline")
    torch_compile: bool = True
    torch_compile_cache: bool = True
    torch_compile_cache_dir: Path = Path(".cache/torch_compile")

    # --- ML / content (from YAML; env vars intentionally not supported) ---

    @property
    def use_controlnet(self) -> bool:
        return content_config.get_use_controlnet()

    @property
    def controlnet_model(self) -> str:
        return content_config.get_controlnet_model()

    @property
    def inpaint_model(self) -> str:
        return content_config.get_default_inpaint_model()

    @property
    def extra_clothes_model(self) -> str:
        return content_config.get_extra_clothes_model()

    @property
    def segformer_model(self) -> str:
        return content_config.get_segformer_model()

    @property
    def detection_threshold(self) -> float:
        return content_config.get_detection_threshold()

    @property
    def pose_keypoint_threshold(self) -> float:
        return content_config.get_pose_keypoint_threshold()

    @property
    def pose_mode(self) -> str:
        return content_config.get_pose_mode()

    @property
    def hand_protect(self) -> bool:
        return content_config.get_hand_protect()

    @property
    def hand_padding_ratio(self) -> float:
        return content_config.get_hand_padding_ratio()

    @property
    def inpaint_steps(self) -> int:
        return content_config.get_inpaint_steps()

    @property
    def guidance_scale(self) -> float:
        return content_config.get_guidance_scale()

    @property
    def inference_size(self) -> int:
        return content_config.get_inference_size()

    @property
    def compile_inpaint_size(self) -> int:
        """Fixed square size for every inpaint pass (keeps torch.compile graphs stable)."""
        from outfit_studio.constants import LATENT_ALIGN, MIN_LATENT_SIDE

        aligned = self.inference_size // LATENT_ALIGN * LATENT_ALIGN
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
