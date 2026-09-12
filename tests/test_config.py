from pathlib import Path

import pytest

from llm_delegator.config import Settings


def test_settings_load_active_models_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "models.toml"
    config_path.write_text(
        '[models]\nactive = ["z-ai/glm-5.3-flash"]\n', encoding="utf-8"
    )
    monkeypatch.setenv("LLM_DELEGATOR_CONFIG", str(config_path))

    settings = Settings.from_env()

    assert settings.active_models == frozenset({"z-ai/glm-5.3-flash"})


def test_settings_reject_invalid_active_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "models.toml"
    config_path.write_text('[models]\nactive = "all"\n', encoding="utf-8")
    monkeypatch.setenv("LLM_DELEGATOR_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="models.active as a list of model IDs"):
        Settings.from_env()
