import json

import httpx
import pytest

from llm_delegator.models import DelegationRequest
from llm_delegator.providers.deepseek import DeepSeekProvider


@pytest.mark.asyncio
async def test_deepseek_adapter_uses_responses_api_and_returns_only_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.deepseek.com/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["reasoning"] == {"effort": "high"}
        assert "selected.py" in payload["input"]
        assert "ACCEPTANCE CRITERIA\n1. Identify the single issue." in payload["input"]
        assert "CONSTRAINTS\n- Do not propose unrelated changes." in payload["input"]
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "hidden"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "final"}],
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 2},
                },
            },
        )

    provider = DeepSeekProvider(transport=httpx.MockTransport(handler))
    result = await provider.delegate(
        DelegationRequest(
            task="Review the selected file for one concrete defect.",
            task_kind="routine_review",
            complexity="low",
            acceptance_criteria=("Identify the single issue.",),
            constraints=("Do not propose unrelated changes.",),
            model="pro",
        ),
        (("selected.py", "print('hello')\n"),),
    )

    assert result.content == "final"
    assert "hidden" not in result.content
    assert result.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_deepseek_adapter_rejects_inactive_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = DeepSeekProvider(
        active_models=frozenset({("deepseek", "deepseek-v4-pro")})
    )

    with pytest.raises(
        ValueError, match="Model is not active: deepseek/deepseek-v4-flash"
    ):
        provider._model_name("auto", "low")
