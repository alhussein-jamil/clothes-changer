"""Gradio Blocks layout and event wiring."""

from __future__ import annotations

import gradio as gr
from gradio_imageslider import ImageSlider

from outfit_studio.constants import DEFAULT_SEED, SEED_MAX
from outfit_studio.content_config import (
    get_app_name,
    get_default_negative_prompt,
    get_default_prompt,
)
from outfit_studio.ui.header import build_header_html
from outfit_studio.ui.theme import (
    CLOTHES_COLOR,
    CUSTOM_CSS,
    EDITOR_CANVAS_SIZE,
    PERSON_COLOR,
    UI,
)


def build_ui(app) -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title=get_app_name(),
        theme="Maani/MonoNeo",
    ) as demo:
        gr.HTML(build_header_html(app.settings), elem_id="app-header")

        with gr.Tabs() as main_tabs:
            with gr.Tab("Generate"):
                with gr.Row():
                    user_info = gr.Textbox(show_label=False, interactive=False)
                    credits_info = gr.Textbox(show_label=False, interactive=False)
                    if app.settings.require_auth:
                        gr.Button("Logout", link="/auth/logout", size="sm")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        input_image = gr.ImageEditor(
                            label="Input Image",
                            type="pil",
                            format="png",
                            interactive=True,
                            image_mode="RGBA",
                            layers=False,
                            canvas_size=EDITOR_CANVAS_SIZE,
                            brush=gr.Brush(
                                colors=[f"rgba{PERSON_COLOR}", f"rgba{CLOTHES_COLOR}"],
                                color_mode="fixed",
                                default_color=f"rgba{CLOTHES_COLOR}",
                            ),
                        )
                        resegment_btn = gr.Button("Redo Clothes Segmentation", size="sm")
                        clean_source = gr.State(value=None)
                        segment_key = gr.State(value=None)
                        segment_masks = gr.State(value=None)
                        suppress_upload_hook = gr.State(value=False)
                        pending_editor = gr.State(value=None)
                        debug_session_dir = gr.State(value=None)
                        debug_status = gr.Markdown(visible=False)

                    with gr.Column(scale=1):
                        result = ImageSlider(
                            label="Before / After",
                            interactive=False,
                        )
                        use_as_input = gr.Button("Use as Input", visible=False, size="sm")
                        with gr.Row():
                            stop_btn = gr.Button(
                                "Stop", variant="stop", size="lg", interactive=False
                            )
                            generate_btn = gr.Button("Generate", variant="primary", size="lg")
                        action_buttons = [generate_btn, stop_btn, resegment_btn]
                        if app.examples:
                            examples = gr.Examples(
                                examples=app.examples,
                                inputs=input_image,
                                label="Examples",
                            )
                        else:
                            examples = None
                        example_index = gr.State(value=None)

                user_prompt_addon = gr.Textbox(
                    label="Additional description (optional)",
                    placeholder="e.g. red linen dress, casual summer outfit",
                    lines=2,
                    visible=True,
                )

                with gr.Accordion("Model & prompts", open=True, visible=False) as admin_settings:
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=app.model_choices,
                            value=app.default_model,
                            label="Model",
                        )
                        use_controlnet = gr.Checkbox(
                            label="Pose ControlNet",
                            value=app.settings.content.use_controlnet,
                        )
                        reload_btn = gr.Button("↻ Models")
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=UI.PROMPT_LINES,
                        value=get_default_prompt(),
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        lines=UI.NEGATIVE_PROMPT_LINES,
                        value=get_default_negative_prompt(),
                    )

                with gr.Accordion("Advanced", open=False, visible=False) as advanced_settings:
                    with gr.Row():
                        steps = gr.Slider(
                            UI.STEPS_SLIDER_MIN,
                            UI.STEPS_SLIDER_MAX,
                            value=app.settings.content.steps,
                            step=1,
                            label="Steps",
                        )
                        guidance = gr.Slider(
                            UI.CFG_SLIDER_MIN,
                            UI.CFG_SLIDER_MAX,
                            value=app.settings.content.guidance_scale,
                            step=UI.CFG_SLIDER_STEP,
                            label="CFG",
                        )
                        seed = gr.Slider(0, SEED_MAX, value=DEFAULT_SEED, step=1, label="Seed")
                        random_seed = gr.Checkbox(label="Random seed", value=True)

            with gr.Tab("History"):
                history_gallery = gr.Gallery(
                    value=app.history_gallery_value,
                    label="Your generations",
                    columns=UI.HISTORY_GALLERY_COLUMNS,
                    height=UI.HISTORY_GALLERY_HEIGHT,
                    object_fit="contain",
                    allow_preview=True,
                    show_download_button=True,
                )
                history_refresh = gr.Button("Refresh history")

        with gr.Accordion("Admin", visible=False, open=False) as admin_panel:
            gr.Markdown("### User management")
            users_df = gr.Dataframe(
                headers=["id", "username", "credits", "admin"],
                interactive=False,
            )
            gr.Button("Refresh").click(app.list_users_table, None, users_df)
            with gr.Row():
                admin_user = gr.Textbox(label="Username")
                admin_credits = gr.Number(
                    label="Credits",
                    value=UI.DEFAULT_ADMIN_CREDITS_INPUT,
                    minimum=0,
                )
                admin_set = gr.Button("Set credits")
            admin_msg = gr.Textbox(label="Result", interactive=False)
            admin_set.click(app.update_credits, [admin_user, admin_credits], admin_msg)

        # Event wiring (Admin is not a hidden Tab — breaks Gradio 5 mount as Tab)
        demo.load(app._admin_panel_boot, None, [admin_panel, users_df])
        demo.load(app._user_label, None, user_info)
        demo.load(app._credits_label, None, credits_info)
        demo.load(
            app._generate_tab_boot,
            None,
            [admin_settings, advanced_settings, user_prompt_addon, debug_status],
        )
        demo.load(app.history_gallery_value, None, history_gallery)
        demo.load(app._refresh_action_buttons, None, action_buttons)
        preload_timer = gr.Timer(value=1, active=True)
        preload_timer.tick(app._refresh_action_buttons, None, action_buttons)
        main_tabs.select(app._load_history_on_tab, None, history_gallery)

        history_refresh.click(app.history_images, None, history_gallery)

        stop_btn.click(app._request_stop, None, None)

        input_image.upload(
            app.sync_clean_source,
            inputs=[input_image, clean_source, segment_key],
            outputs=clean_source,
        ).success(
            app._begin_operation,
            None,
            action_buttons,
        ).then(
            app.prepare_upload_segment,
            inputs=[
                input_image,
                segment_key,
                clean_source,
                suppress_upload_hook,
                debug_session_dir,
                segment_masks,
            ],
            outputs=[
                pending_editor,
                clean_source,
                segment_key,
                suppress_upload_hook,
                debug_session_dir,
                segment_masks,
            ],
        ).then(
            app._apply_pending_editor,
            pending_editor,
            input_image,
        ).then(
            app._clear_pending_editor,
            pending_editor,
            pending_editor,
        ).then(
            app._end_operation,
            None,
            action_buttons,
        )
        input_image.clear(
            app.clear_editor_state,
            None,
            [
                clean_source,
                segment_key,
                suppress_upload_hook,
                debug_session_dir,
                segment_masks,
            ],
        )
        resegment_btn.click(
            app._begin_operation,
            None,
            action_buttons,
        ).then(
            app.resegment,
            [input_image, clean_source, segment_key, debug_session_dir, segment_masks],
            [
                pending_editor,
                clean_source,
                segment_key,
                suppress_upload_hook,
                debug_session_dir,
                segment_masks,
            ],
        ).then(
            app._apply_pending_editor,
            pending_editor,
            input_image,
        ).then(
            app._clear_pending_editor,
            pending_editor,
            pending_editor,
        ).then(
            app._debug_status_update,
            debug_session_dir,
            debug_status,
        ).then(
            app._end_operation,
            None,
            action_buttons,
        )

        def reload_models() -> gr.Dropdown:
            app._refresh_models()
            return gr.Dropdown(
                choices=app.model_choices,
                value=app.default_model,
            )

        reload_btn.click(reload_models, None, model_dropdown)

        random_seed.change(
            lambda r: gr.update(interactive=not r),
            random_seed,
            seed,
        )

        generate_btn.click(
            app._begin_operation,
            None,
            action_buttons,
        ).then(
            lambda: gr.update(value=None),
            None,
            result,
        ).then(
            app.generate,
            [
                input_image,
                clean_source,
                segment_key,
                prompt,
                negative_prompt,
                model_dropdown,
                use_controlnet,
                steps,
                guidance,
                seed,
                random_seed,
                debug_session_dir,
                user_prompt_addon,
            ],
            [result, seed, debug_session_dir],
        ).then(
            app._debug_status_update,
            debug_session_dir,
            debug_status,
        ).then(lambda: gr.update(visible=True), None, use_as_input).then(
            app._credits_label, None, credits_info
        ).then(app.history_images, None, history_gallery).then(
            app._end_operation,
            None,
            action_buttons,
        )

        use_as_input.click(
            app._begin_operation,
            None,
            action_buttons,
        ).then(
            app.use_result_as_input,
            [result, debug_session_dir],
            [
                input_image,
                clean_source,
                segment_key,
                suppress_upload_hook,
                debug_session_dir,
                segment_masks,
            ],
        ).then(
            app._end_operation,
            None,
            action_buttons,
        )

        if examples is not None:
            examples.dataset.select(
                app._store_example_index,
                None,
                example_index,
            ).then(
                app._begin_operation,
                None,
                action_buttons,
            ).then(
                app.load_example_after_select,
                inputs=[input_image, example_index, debug_session_dir],
                outputs=[
                    input_image,
                    clean_source,
                    segment_key,
                    suppress_upload_hook,
                    debug_session_dir,
                    segment_masks,
                ],
            ).then(
                app._debug_status_update,
                debug_session_dir,
                debug_status,
            ).then(
                app._end_operation,
                None,
                action_buttons,
            )

    return demo
