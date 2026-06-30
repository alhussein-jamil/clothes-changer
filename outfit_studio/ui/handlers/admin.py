"""Admin panel handlers for GradioApp."""

from __future__ import annotations

import gradio as gr


class AdminHandlersMixin:
    def _debug_status_update(
        self, debug_session_dir: str | None, request: gr.Request
    ) -> gr.Markdown:
        if not self.is_admin(request) or not self.settings.pipeline_debug:
            return gr.update(visible=False)
        if debug_session_dir:
            return gr.update(
                value=f"**Debug artifacts:** `{debug_session_dir}`",
                visible=True,
            )
        return gr.update(
            value="**Debug mode:** waiting for segmentation or generation…",
            visible=True,
        )

    def list_users_table(self, request: gr.Request) -> list[list]:
        if not self.is_admin(request):
            raise gr.Error("Admin access required")
        return [
            [u.id, u.username, u.credits, "Yes" if u.is_admin else "No"]
            for u in self.db.list_users()
        ]

    def update_credits(self, username: str, credits: int, request: gr.Request) -> str:
        if not self.is_admin(request):
            raise gr.Error("Admin access required")
        if not username or credits < 0:
            return "Invalid input"
        result = "Updated" if self.db.set_credits(username, int(credits)) else "Failed"
        return result

    def _admin_panel_boot(self, request: gr.Request) -> tuple[dict, list[list] | dict]:
        """Show admin accordion and populate users table for admins only."""
        if self.is_admin(request):
            return gr.update(visible=True), self.list_users_table(request)
        return gr.update(visible=False), gr.skip()

    def _generate_tab_boot(self, request: gr.Request) -> tuple[dict, dict, dict, dict]:
        """Show admin-only controls and debug output only for administrators."""
        admin = self.is_admin(request)
        show_debug = admin and self.settings.pipeline_debug
        return (
            gr.update(visible=admin),
            gr.update(visible=admin),
            gr.update(visible=not admin),
            gr.update(visible=show_debug),
        )
