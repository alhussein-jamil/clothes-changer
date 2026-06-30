"""Gradio web interface."""

from __future__ import annotations

import logging

import gradio as gr

from outfit_studio.config import Settings, get_settings, validate_deployment_settings
from outfit_studio.db.database import Database
from outfit_studio.ml.inpainter import get_inpaint_engine
from outfit_studio.ml.pipeline import get_pipeline
from outfit_studio.ui.handlers.admin import AdminHandlersMixin
from outfit_studio.ui.handlers.generation import GenerationHandlersMixin
from outfit_studio.ui.handlers.history import HistoryHandlersMixin
from outfit_studio.ui.handlers.images import ImageHandlersMixin
from outfit_studio.ui.handlers.operations import OperationHandlersMixin
from outfit_studio.ui.handlers.segmentation import SegmentationHandlersMixin
from outfit_studio.ui.handlers.session import SessionHandlersMixin
from outfit_studio.ui.header import build_header_html
from outfit_studio.ui.launch import collect_allowed_paths, gradio_launch_kwargs
from outfit_studio.ui.tabs.layout import build_ui

logger = logging.getLogger(__name__)

__all__ = ["GradioApp", "build_header_html"]


class GradioApp(
    SessionHandlersMixin,
    ImageHandlersMixin,
    SegmentationHandlersMixin,
    GenerationHandlersMixin,
    HistoryHandlersMixin,
    AdminHandlersMixin,
    OperationHandlersMixin,
):
    """Outfit Studio Gradio UI."""

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = db or Database()
        self.pipeline = get_pipeline()
        self.examples = self._load_examples()
        self._refresh_models()
        logger.info(
            "GradioApp initialized (%d examples, %d models)",
            len(self.examples),
            len(self.model_ids),
        )

    def _refresh_models(self) -> None:
        engine = get_inpaint_engine()
        engine.invalidate_model_list_cache()
        models = engine.list_models()
        self.models = models
        self.model_ids = [m["id"] for m in models]
        self.model_choices = [(f"{m['name']} ({m['arch']})", m["id"]) for m in models]
        self.default_model = engine.default_model_id()

    def create_ui(self) -> gr.Blocks:
        return build_ui(self)

    def launch(self) -> None:
        self.settings.ensure_dirs()
        if self.settings.allow_bootstrap_admin and not self.db.user_exists(
            self.settings.default_admin
        ):
            try:
                self.db.register_user(
                    self.settings.default_admin,
                    self.settings.default_password,
                    credits=self.settings.default_credits,
                    is_admin=True,
                )
                logger.info("Bootstrapped default admin %r", self.settings.default_admin)
            except Exception as e:
                logger.warning("Admin bootstrap: %s", e)

        logger.info("Building Gradio UI …")
        static_dir = self.settings.resolved_static_dir
        if static_dir.is_dir():
            gr.set_static_paths(paths=[static_dir])
        demo = self.create_ui()
        demo.queue(default_concurrency_limit=1)
        allowed = collect_allowed_paths(self.settings)
        logger.info(
            "Starting server %s:%d (share=%s, auth=%s, %d allowed paths)",
            self.settings.host,
            self.settings.port,
            self.settings.enable_sharing,
            self.settings.require_auth,
            len(allowed),
        )
        get_inpaint_engine().start_background_preload()
        if self.settings.require_auth:
            validate_deployment_settings(self.settings)
            from outfit_studio.auth.app import launch_with_fastapi

            launch_with_fastapi(self, demo)
            return

        demo.launch(**gradio_launch_kwargs(self.settings, allowed))
