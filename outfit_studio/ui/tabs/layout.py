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
from outfit_studio.ui.tabs.events import register_events
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

        register_events(
            app,
            demo,
            main_tabs=main_tabs,
            user_info=user_info,
            credits_info=credits_info,
            input_image=input_image,
            resegment_btn=resegment_btn,
            clean_source=clean_source,
            segment_key=segment_key,
            segment_masks=segment_masks,
            suppress_upload_hook=suppress_upload_hook,
            pending_editor=pending_editor,
            debug_session_dir=debug_session_dir,
            debug_status=debug_status,
            result=result,
            use_as_input=use_as_input,
            stop_btn=stop_btn,
            generate_btn=generate_btn,
            action_buttons=action_buttons,
            examples=examples,
            example_index=example_index,
            user_prompt_addon=user_prompt_addon,
            admin_settings=admin_settings,
            advanced_settings=advanced_settings,
            model_dropdown=model_dropdown,
            use_controlnet=use_controlnet,
            reload_btn=reload_btn,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            seed=seed,
            random_seed=random_seed,
            history_gallery=history_gallery,
            history_refresh=history_refresh,
            admin_panel=admin_panel,
            users_df=users_df,
            admin_user=admin_user,
            admin_credits=admin_credits,
            admin_set=admin_set,
            admin_msg=admin_msg,
        )

    return demo
