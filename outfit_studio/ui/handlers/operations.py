"""Long-running operation controls for GradioApp."""

from __future__ import annotations

import gradio as gr

from outfit_studio.ml.inpainter import get_inpaint_engine
from outfit_studio.ui.operation_control import begin_operation, request_stop, session_hash_from


class OperationHandlersMixin:
    @staticmethod
    def _action_button_updates(*, busy: bool) -> tuple[dict, dict, dict]:
        return (
            gr.update(interactive=not busy),
            gr.update(interactive=busy),
            gr.update(interactive=not busy),
        )

    def _refresh_action_buttons(self, *, busy: bool | None = None) -> tuple[dict, dict, dict]:
        if busy is None:
            busy = get_inpaint_engine().is_preparing()
        return self._action_button_updates(busy=busy)

    def _begin_operation(self, request: gr.Request) -> tuple[dict, dict, dict]:
        begin_operation(session_hash_from(request))
        get_inpaint_engine().clear_work_abort()
        return self._action_button_updates(busy=True)

    def _end_operation(self) -> tuple[dict, dict, dict]:
        return self._refresh_action_buttons()

    def _request_stop(self, request: gr.Request) -> None:
        request_stop(session_hash_from(request))
        get_inpaint_engine().request_abort()
