"""FastAPI application with Gradio mounted and production security."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request

from outfit_studio.auth.routes import register_auth_routes
from outfit_studio.auth.security import AuthRedirectMiddleware, SecurityHeadersMiddleware
from outfit_studio.auth.session import SessionManager

if TYPE_CHECKING:
    from outfit_studio.ui.gradio_app import GradioApp

logger = logging.getLogger(__name__)


def create_fastapi_app(gradio_app: GradioApp, demo: gr.Blocks) -> FastAPI:
    settings = gradio_app.settings
    sessions = SessionManager(settings)

    def resolve_username(request: Request) -> str | None:
        return sessions.username_from_request(request)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AuthRedirectMiddleware, settings=settings, resolve_username=resolve_username)
    register_auth_routes(app, gradio_app.db, settings, sessions)

    auth_dependency = resolve_username if settings.require_auth else None
    launch_kwargs: dict = {
        "server_name": settings.host,
        "server_port": settings.port,
        "share": settings.enable_sharing,
        "allowed_paths": gradio_app._allowed_paths(),
    }
    favicon = settings.resolved_favicon_path if settings.resolved_favicon_path.is_file() else None
    if favicon:
        launch_kwargs["favicon_path"] = str(favicon)

    gr.mount_gradio_app(
        app,
        demo,
        path="/",
        auth_dependency=auth_dependency,
        favicon_path=launch_kwargs.get("favicon_path"),
        allowed_paths=launch_kwargs["allowed_paths"],
        show_error=settings.debug,
    )
    return app


def launch_with_fastapi(gradio_app: GradioApp, demo: gr.Blocks) -> None:
    settings = gradio_app.settings
    app = create_fastapi_app(gradio_app, demo)
    logger.info(
        "Starting FastAPI+Gradio %s:%d (auth=%s)",
        settings.host,
        settings.port,
        settings.require_auth,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips or "*",
    )
