"""OpenRouter Chat Completions API adapter."""

import os
from typing import Any

import httpx

from llm_delegator.config import ModelIdentity
from llm_delegator.models import DelegationRequest, DelegationResult
from llm_delegator.providers.prompt import instructions, user_input

_DEFAULT_MODELS = {
    "flash": "z-ai/glm-5.3-flash",
    "pro": "z-ai/glm-5.3",
}
_REASONING_EFFORTS = {
    "none": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "max": "max",
}


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        *,
        timeout_seconds: float = 180,
        transport: httpx.AsyncBaseTransport | None = None,
        active_models: frozenset[ModelIdentity] | None = None,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self._api_key = api_key
        self._base_url = os.getenv(
            "LLM_DELEGATOR_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport
        self._active_models = active_models

    def _model_name(self, requested: str, complexity: str) -> str:
        if complexity == "low":
            requested = "flash"
        elif requested == "auto":
            requested = "pro"
        env_name = f"LLM_DELEGATOR_OPENROUTER_MODEL_{requested.upper()}"
        model = os.getenv(env_name, _DEFAULT_MODELS.get(requested, requested))
        if (
            self._active_models is not None
            and (
                self.name,
                model,
            )
            not in self._active_models
        ):
            raise ValueError(f"Model is not active: {self.name}/{model}")
        return model

    async def delegate(
        self,
        request: DelegationRequest,
        files: tuple[tuple[str, str], ...],
    ) -> DelegationResult:
        model = self._model_name(request.model, request.complexity)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions(request.output_format)},
                {"role": "user", "content": user_input(request, files)},
            ],
            "reasoning": {"effort": _REASONING_EFFORTS[request.reasoning_effort]},
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions", headers=headers, json=payload
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = _safe_error(response)
            raise RuntimeError(
                f"OpenRouter API returned HTTP {response.status_code}: {detail}"
            ) from error

        body = response.json()
        content = _final_text(body)
        if not content:
            raise RuntimeError("OpenRouter response contained no final text")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        normalized_usage = {
            output_name: int(usage[input_name])
            for input_name, output_name in {
                "prompt_tokens": "input_tokens",
                "completion_tokens": "output_tokens",
                "total_tokens": "total_tokens",
            }.items()
            if isinstance(usage.get(input_name), int)
        }
        return DelegationResult(
            provider=self.name,
            model=model,
            content=content,
            files_read=tuple(name for name, _ in files),
            usage=normalized_usage,
        )


def _final_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return ""
    content = choice["message"].get("content")
    return content.strip() if isinstance(content, str) else ""


def _safe_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:500]
    return "request failed"
