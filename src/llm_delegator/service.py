"""Delegation orchestration independent of MCP and CLI transports."""

from llm_delegator.config import Settings
from llm_delegator.models import DelegationRequest, DelegationResult
from llm_delegator.providers import DeepSeekProvider, ModelProvider
from llm_delegator.workspace import read_workspace_files, resolve_workspace_root


class DelegationService:
    def __init__(
        self,
        settings: Settings | None = None,
        providers: dict[str, ModelProvider] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._providers = providers

    def _provider(self, name: str) -> ModelProvider:
        if self._providers is not None:
            try:
                return self._providers[name]
            except KeyError as error:
                raise ValueError(f"Unknown provider: {name}") from error
        if name == "deepseek":
            return DeepSeekProvider(
                timeout_seconds=self.settings.request_timeout_seconds
            )
        raise ValueError(f"Unknown provider: {name}")

    async def delegate(self, request: DelegationRequest) -> DelegationResult:
        if not request.task.strip():
            raise ValueError("task must not be empty")
        if len(request.context) > self.settings.max_context_chars:
            raise ValueError(
                f"context exceeds the {self.settings.max_context_chars}-character limit"
            )
        if request.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if request.max_output_tokens > 100_000:
            raise ValueError("max_output_tokens must not exceed 100000")

        root = resolve_workspace_root(request.workspace_root, self.settings)
        files = read_workspace_files(root, request.files, self.settings)
        return await self._provider(request.provider).delegate(request, files)
