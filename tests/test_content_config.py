from outfit_studio.content_config import (
    clear_content_config_cache,
    get_app_name,
    get_default_inpaint_model,
    get_default_prompt,
    get_tagline,
)


def test_content_config_defaults(tmp_path, monkeypatch):
    default = tmp_path / "content.default.yaml"
    default.write_text(
        "app:\n  name: Outfit Studio\n  tagline: AI outfit inpainting\n"
        "prompts:\n  default: high quality clothing\n",
        encoding="utf-8",
    )
    missing = tmp_path / "content.local.yaml"
    monkeypatch.setattr("outfit_studio.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("outfit_studio.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("outfit_studio.content_config._LOCAL_FILE", missing)
    clear_content_config_cache()

    assert get_app_name() == "Outfit Studio"
    assert get_tagline() == "AI outfit inpainting"
    assert "clothing" in get_default_prompt().lower()


def test_content_config_local_override(tmp_path, monkeypatch):
    default = tmp_path / "content.default.yaml"
    local = tmp_path / "content.local.yaml"
    default.write_text(
        "app:\n  name: Default\nprompts:\n  default: safe prompt\n",
        encoding="utf-8",
    )
    local.write_text(
        "app:\n  name: Local Override\nprompts:\n  default: custom prompt\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("outfit_studio.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("outfit_studio.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("outfit_studio.content_config._LOCAL_FILE", local)
    clear_content_config_cache()

    assert get_app_name() == "Local Override"
    assert get_default_prompt() == "custom prompt"


def test_shipped_default_inpaint_is_hub_model(tmp_path, monkeypatch):
    default = tmp_path / "content.default.yaml"
    default.write_text(
        "models:\n  default_inpaint: runwayml/stable-diffusion-inpainting\n"
        "prompts:\n  default: detailed clothing, fabric texture\n",
        encoding="utf-8",
    )
    missing = tmp_path / "content.local.yaml"
    monkeypatch.setattr("outfit_studio.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("outfit_studio.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("outfit_studio.content_config._LOCAL_FILE", missing)
    clear_content_config_cache()

    assert get_default_inpaint_model() == "runwayml/stable-diffusion-inpainting"
    assert "clothing" in get_default_prompt().lower()


def test_ml_defaults_from_yaml(tmp_path, monkeypatch):
    default = tmp_path / "content.default.yaml"
    default.write_text(
        "models:\n"
        "  default_inpaint: custom.safetensors\n"
        "  human_parser: org/human-parser\n"
        "  controlnet: org/controlnet\n"
        "generation:\n"
        "  use_controlnet: false\n"
        "  steps: 30\n"
        "  guidance_scale: 7.0\n"
        "  inference_size: 640\n"
        "pose:\n"
        "  detection_threshold: 0.4\n"
        "  keypoint_threshold: 0.2\n"
        "  mode: performance\n",
        encoding="utf-8",
    )
    missing = tmp_path / "content.local.yaml"
    monkeypatch.setattr("outfit_studio.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("outfit_studio.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("outfit_studio.content_config._LOCAL_FILE", missing)
    clear_content_config_cache()

    from outfit_studio.content_config import (
        get_controlnet_model,
        get_detection_threshold,
        get_guidance_scale,
        get_human_parser_model,
        get_inference_size,
        get_inpaint_steps,
        get_pose_keypoint_threshold,
        get_pose_mode,
        get_use_controlnet,
    )

    assert get_default_inpaint_model() == "custom.safetensors"
    assert get_human_parser_model() == "org/human-parser"
    assert get_controlnet_model() == "org/controlnet"
    assert get_use_controlnet() is False
    assert get_inpaint_steps() == 30
    assert get_guidance_scale() == 7.0
    assert get_inference_size() == 640
    from outfit_studio.config import get_settings

    get_settings.cache_clear()
    assert get_settings().compile_inpaint_size == 640
    assert get_detection_threshold() == 0.4
    assert get_pose_keypoint_threshold() == 0.2
    assert get_pose_mode() == "performance"


def test_settings_reads_inpaint_from_yaml_not_env(tmp_path, monkeypatch):
    default = tmp_path / "content.default.yaml"
    default.write_text(
        "models:\n  default_inpaint: yaml-model.safetensors\n",
        encoding="utf-8",
    )
    missing = tmp_path / "content.local.yaml"
    monkeypatch.setattr("outfit_studio.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("outfit_studio.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("outfit_studio.content_config._LOCAL_FILE", missing)
    monkeypatch.setenv("OUTFIT_STUDIO_INPAINT_MODEL", "env-model.safetensors")

    from outfit_studio.config import get_settings

    clear_content_config_cache()
    get_settings.cache_clear()
    assert get_settings().inpaint_model == "yaml-model.safetensors"
