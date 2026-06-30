"""History gallery handlers for GradioApp."""

from __future__ import annotations

import gradio as gr

from outfit_studio.ui.theme import UI


class HistoryHandlersMixin:
    def history_images(self, request: gr.Request | None = None) -> list[tuple[str, str]]:
        """Return gallery items as (absolute_path, caption) tuples."""
        username = self._session_username(request)
        if not username and not self.settings.require_auth:
            username = self.settings.default_admin
        if not username:
            return []

        out_dir = self.settings.resolved_output_dir
        show_prompts = self.is_admin(request)
        items: list[tuple[str, str]] = []

        for row in self.db.get_history(username):
            fn = row.get("filename")
            if not fn:
                continue
            path = out_dir / fn
            if not path.is_file():
                continue
            if show_prompts:
                caption = (row.get("prompt") or fn)[: UI.HISTORY_CAPTION_MAX_LEN]
            else:
                caption = "Generation"
            items.append((str(path.resolve()), caption))

        return items[: UI.HISTORY_GALLERY_LIMIT]

    def _load_history_on_tab(self, evt: gr.SelectData, request: gr.Request) -> list | dict:
        """Load gallery when the History tab is opened."""
        if evt.selected and evt.index == 1:
            return self.history_images(request)
        return gr.update()

    def history_gallery_value(self, request: gr.Request | None = None) -> list[tuple[str, str]]:
        """Initial gallery value on page load."""
        return self.history_images(request)
