"""Delegation orchestration independent of MCP and CLI transports."""

from llm_delegator.config import Settings
from llm_delegator.models import DelegationRequest, DelegationResult
from llm_delegator.providers import DeepSeekProvider, ModelProvider
from llm_delegator.workspace import read_workspace_files, resolve_workspace_root

_ALLOWED_TASK_KINDS = {
    "summarize",
    "extract",
    "classify",
    "routine_transform",
    "draft_documentation",
    "draft_tests",
    "routine_code_draft",
    "routine_review",
}


def _validate_handoff_items(
    name: str,
    items: tuple[str, ...],
    *,
    required: bool,
    maximum: int,
) -> None:
    if required and not items:
        raise ValueError(f"{name} must contain at least one item")
    if len(items) > maximum:
        raise ValueError(f"{name} must not contain more than {maximum} items")
    if any(not item.strip() for item in items):
        raise ValueError(f"{name} must not contain blank items")
    if any(len(item) > 2_000 for item in items):
        raise ValueError(f"each {name} item must not exceed 2000 characters")


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
        if request.task_kind not in _ALLOWED_TASK_KINDS:
            raise ValueError(f"unsupported task_kind: {request.task_kind}")
        if request.complexity not in {"low", "medium"}:
            raise ValueError(
                "complexity must be low or medium; complex tasks stay with the primary model"
            )
        _validate_handoff_items(
            "acceptance_criteria",
            request.acceptance_criteria,
            required=True,
            maximum=20,
        )
        _validate_handoff_items("plan", request.plan, required=False, maximum=30)
        _validate_handoff_items(
            "constraints", request.constraints, required=False, maximum=20
        )
        if request.complexity == "medium" and not request.plan:
            raise ValueError("medium-complexity tasks require an ordered plan")
        if len(request.context) > self.settings.max_context_chars:
            raise ValueError(
                f"context exceeds the {self.settings.max_context_chars}-character limit"
            )
        if request.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if request.max_output_tokens > 16_000:
            raise ValueError("max_output_tokens must not exceed 16000")

        root = resolve_workspace_root(request.workspace_root, self.settings)
        files = read_workspace_files(root, request.files, self.settings)
        return await self._provider(request.provider).delegate(request, files)
