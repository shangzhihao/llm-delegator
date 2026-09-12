import json

import httpx
import pytest

from llm_delegator.models import DelegationRequest
from llm_delegator.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "complexity",
        "requested_model",
        "reasoning_effort",
        "expected_effort",
        "expected_model",
    ),
    [
        ("low", "pro", "none", "low", "z-ai/glm-5.3-flash"),
        ("medium", "auto", "medium", "high", "z-ai/glm-5.3"),
    ],
)
async def test_openrouter_adapter_routes_glm_models_and_normalizes_response(
    monkeypatch: pytest.MonkeyPatch,
    complexity: str,
    requested_model: str,
    reasoning_effort: str,
    expected_effort: str,
    expected_model: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == expected_model
        assert payload["reasoning"] == {"effort": expected_effort}
        assert payload["max_tokens"] == 4_000
        assert payload["messages"][0]["role"] == "system"
        assert "selected.py" in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "final",
                            "reasoning": "hidden",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    provider = OpenRouterProvider(transport=httpx.MockTransport(handler))
    result = await provider.delegate(
        DelegationRequest(
            task="Review the selected file for one concrete defect.",
            task_kind="routine_review",
            complexity=complexity,  # type: ignore[arg-type]
            acceptance_criteria=("Identify the single issue.",),
            plan=("Inspect the selected file.",) if complexity == "medium" else (),
            model=requested_model,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        ),
        (("selected.py", "print('hello')\n"),),
    )

    assert result.provider == "openrouter"
    assert result.model == expected_model
    assert result.content == "final"
    assert "hidden" not in result.content
    assert result.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_openrouter_adapter_rejects_inactive_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenRouterProvider(
        active_models=frozenset({("z.ai", "z-ai/glm-5.3-flash")})
    )

    with pytest.raises(
        ValueError, match="Model is not active: openrouter/z-ai/glm-5.3-flash"
    ):
        provider._model_name("auto", "low")
