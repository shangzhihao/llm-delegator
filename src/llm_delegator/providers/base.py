"""Provider interface."""

from typing import Protocol

from llm_delegator.models import DelegationRequest, DelegationResult


class ModelProvider(Protocol):
    name: str

    async def delegate(
        self,
        request: DelegationRequest,
        files: tuple[tuple[str, str], ...],
    ) -> DelegationResult: ...
