"""DeepSeek Responses API adapter."""

import os
from typing import Any

import httpx

from llm_delegator.models import DelegationRequest, DelegationResult
from llm_delegator.providers.prompt import instructions, user_input

_DEFAULT_MODELS = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        *,
        timeout_seconds: float = 180,
        transport: httpx.AsyncBaseTransport | None = None,
        active_models: frozenset[str] | None = None,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self._api_key = api_key
        self._base_url = os.getenv(
            "LLM_DELEGATOR_DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport
        self._active_models = active_models

    def _model_name(self, requested: str, complexity: str) -> str:
        if complexity == "low":
            requested = "flash"
        elif requested == "auto":
            requested = "pro"
        env_name = f"LLM_DELEGATOR_DEEPSEEK_MODEL_{requested.upper()}"
        model = os.getenv(env_name, _DEFAULT_MODELS.get(requested, requested))
        if self._active_models is not None and model not in self._active_models:
            raise ValueError(f"Model is not active: {model}")
        return model

    async def delegate(
        self,
        request: DelegationRequest,
        files: tuple[tuple[str, str], ...],
    ) -> DelegationResult:
        model = self._model_name(request.model, request.complexity)
        payload = {
            "model": model,
            "instructions": instructions(request.output_format),
            "input": user_input(request, files),
            "reasoning": {"effort": request.reasoning_effort},
            "max_output_tokens": request.max_output_tokens,
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
                f"{self._base_url}/responses", headers=headers, json=payload
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = _safe_error(response)
            raise RuntimeError(
                f"DeepSeek API returned HTTP {response.status_code}: {detail}"
            ) from error

        body = response.json()
        if body.get("status") == "failed":
            raise RuntimeError(
                f"DeepSeek response failed: {body.get('error', 'unknown error')}"
            )
        content = _final_text(body)
        if not content:
            raise RuntimeError("DeepSeek response contained no final text")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        normalized_usage = {
            key: int(value)
            for key, value in usage.items()
            if key in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(value, int)
        }
        return DelegationResult(
            provider=self.name,
            model=model,
            content=content,
            files_read=tuple(name for name, _ in files),
            usage=normalized_usage,
        )


def _final_text(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


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
