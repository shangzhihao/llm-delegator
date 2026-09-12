from pathlib import Path

import pytest

from llm_delegator.config import Settings


def test_settings_load_active_models_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "models.toml"
    config_path.write_text(
        """[models]
active = [
    { provider = "openrouter", model = "shared/glm-5.3" },
    { provider = "z.ai", model = "shared/glm-5.3" },
]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_DELEGATOR_CONFIG", str(config_path))

    settings = Settings.from_env()

    assert settings.active_models == frozenset(
        {
            ("openrouter", "shared/glm-5.3"),
            ("z.ai", "shared/glm-5.3"),
        }
    )


def test_settings_reject_invalid_active_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "models.toml"
    config_path.write_text('[models]\nactive = ["z-ai/glm-5.3"]\n', encoding="utf-8")
    monkeypatch.setenv("LLM_DELEGATOR_CONFIG", str(config_path))

    with pytest.raises(TypeError, match="list of provider/model tables"):
        Settings.from_env()
