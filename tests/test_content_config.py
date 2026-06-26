from clothes_changer.content_config import (
    clear_content_config_cache,
    get_app_name,
    get_default_prompt,
    get_title_html,
)


def test_content_config_defaults(tmp_path, monkeypatch):
    default = tmp_path / "content.default.yaml"
    default.write_text(
        "app:\n  name: Clothes Changer\n"
        "prompts:\n  default: high quality clothing\n"
        "ui:\n  title_html: '<h1>Clothes Changer</h1>'\n",
        encoding="utf-8",
    )
    missing = tmp_path / "content.local.yaml"
    monkeypatch.setattr("clothes_changer.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("clothes_changer.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("clothes_changer.content_config._LOCAL_FILE", missing)
    clear_content_config_cache()

    assert get_app_name() == "Clothes Changer"
    assert "clothing" in get_default_prompt().lower()
    assert "Clothes Changer" in get_title_html()


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
    monkeypatch.setattr("clothes_changer.content_config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("clothes_changer.content_config._DEFAULT_FILE", default)
    monkeypatch.setattr("clothes_changer.content_config._LOCAL_FILE", local)
    clear_content_config_cache()

    assert get_app_name() == "Local Override"
    assert get_default_prompt() == "custom prompt"
