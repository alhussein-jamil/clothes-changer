"""Shared Gradio launch helpers."""

from __future__ import annotations

import gradio as gr

from outfit_studio.config import Settings


def collect_allowed_paths(settings: Settings) -> list[str]:
    paths = {
        settings.resolved_output_dir,
        settings.resolved_models_dir,
        settings.resolved_examples_dir,
        settings.resolved_static_dir,
    }
    logo = settings.resolved_logo_path
    paths.add(logo)
    favicon = settings.resolved_favicon_path
    if favicon.is_file():
        paths.add(favicon)
    return [str(p.resolve()) for p in paths]


def gradio_launch_kwargs(settings: Settings, allowed_paths: list[str]) -> dict:
    favicon = settings.resolved_favicon_path if settings.resolved_favicon_path.is_file() else None
    return {
        "server_name": settings.host,
        "server_port": settings.port,
        "share": settings.enable_sharing,
        "favicon_path": str(favicon) if favicon else None,
        "allowed_paths": allowed_paths,
    }


def mount_gradio_on_fastapi(
    app,
    demo: gr.Blocks,
    settings: Settings,
    *,
    auth_dependency,
    allowed_paths: list[str],
) -> None:
    favicon = settings.resolved_favicon_path if settings.resolved_favicon_path.is_file() else None
    gr.mount_gradio_app(
        app,
        demo,
        path="/",
        auth_dependency=auth_dependency,
        favicon_path=str(favicon) if favicon else None,
        allowed_paths=allowed_paths,
        show_error=settings.debug,
    )
