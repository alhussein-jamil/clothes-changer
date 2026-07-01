"""Gradio event wiring for the main UI."""

from __future__ import annotations

import gradio as gr


def register_events(
    app,
    demo: gr.Blocks,
    *,
    main_tabs: gr.Tabs,
    user_info: gr.Textbox,
    credits_info: gr.Textbox,
    input_image: gr.ImageEditor,
    resegment_btn: gr.Button,
    clean_source: gr.State,
    segment_key: gr.State,
    segment_masks: gr.State,
    suppress_upload_hook: gr.State,
    pending_editor: gr.State,
    debug_session_dir: gr.State,
    debug_status: gr.Markdown,
    result: gr.ImageSlider,
    use_as_input: gr.Button,
    stop_btn: gr.Button,
    generate_btn: gr.Button,
    action_buttons: list,
    examples,
    example_index: gr.State,
    user_prompt_addon: gr.Textbox,
    admin_settings: gr.Accordion,
    advanced_settings: gr.Accordion,
    model_dropdown: gr.Dropdown,
    use_controlnet: gr.Checkbox,
    reload_btn: gr.Button,
    prompt: gr.Textbox,
    negative_prompt: gr.Textbox,
    steps: gr.Slider,
    guidance: gr.Slider,
    seed: gr.Slider,
    random_seed: gr.Checkbox,
    history_gallery: gr.Gallery,
    history_refresh: gr.Button,
    admin_panel: gr.Accordion,
    users_df: gr.Dataframe,
    admin_user: gr.Textbox,
    admin_credits: gr.Number,
    admin_set: gr.Button,
    admin_msg: gr.Textbox,
) -> None:
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
            segment_masks,
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
    ).then(
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
