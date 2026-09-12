from pathlib import Path

import pytest

from llm_delegator.config import Settings
from llm_delegator.models import DelegationRequest, DelegationResult
from llm_delegator.service import DelegationService


class RecordingProvider:
    name = "fake"

    def __init__(self) -> None:
        self.files: tuple[tuple[str, str], ...] = ()

    async def delegate(
        self,
        request: DelegationRequest,
        files: tuple[tuple[str, str], ...],
    ) -> DelegationResult:
        self.files = files
        return DelegationResult(provider=self.name, model=request.model, content="done")


@pytest.mark.asyncio
async def test_service_passes_only_selected_files(tmp_path: Path) -> None:
    (tmp_path / "selected.txt").write_text("selected", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")
    provider = RecordingProvider()
    service = DelegationService(
        settings=Settings(
            allowed_roots=(tmp_path.resolve(),),
            max_file_bytes=100,
            max_total_file_bytes=200,
            max_context_chars=1_000,
            request_timeout_seconds=10,
        ),
        providers={"fake": provider},
    )

    result = await service.delegate(
        DelegationRequest(
            task="Summarize",
            task_kind="summarize",
            complexity="low",
            provider="fake",
            workspace_root=str(tmp_path),
            files=("selected.txt",),
        )
    )

    assert result.content == "done"
    assert provider.files == (("selected.txt", "selected"),)


@pytest.mark.asyncio
async def test_service_rejects_empty_task(tmp_path: Path) -> None:
    service = DelegationService(
        settings=Settings(
            allowed_roots=(tmp_path.resolve(),),
            max_file_bytes=100,
            max_total_file_bytes=200,
            max_context_chars=1_000,
            request_timeout_seconds=10,
        ),
        providers={},
    )

    with pytest.raises(ValueError, match="task must not be empty"):
        await service.delegate(
            DelegationRequest(
                task="  ",
                task_kind="summarize",
                complexity="low",
                workspace_root=str(tmp_path),
            )
        )


@pytest.mark.asyncio
async def test_service_rejects_non_routine_task_kind(tmp_path: Path) -> None:
    service = DelegationService(
        settings=Settings(
            allowed_roots=(tmp_path.resolve(),),
            max_file_bytes=100,
            max_total_file_bytes=200,
            max_context_chars=1_000,
            request_timeout_seconds=10,
        ),
        providers={},
    )

    with pytest.raises(ValueError, match="unsupported task_kind"):
        await service.delegate(
            DelegationRequest(
                task="Design the system architecture",
                task_kind="architecture",  # type: ignore[arg-type]
                complexity="low",
                workspace_root=str(tmp_path),
            )
        )


@pytest.mark.asyncio
async def test_service_rejects_high_complexity(tmp_path: Path) -> None:
    service = DelegationService(
        settings=Settings(
            allowed_roots=(tmp_path.resolve(),),
            max_file_bytes=100,
            max_total_file_bytes=200,
            max_context_chars=1_000,
            request_timeout_seconds=10,
        ),
        providers={},
    )

    with pytest.raises(ValueError, match="complex tasks stay with the primary model"):
        await service.delegate(
            DelegationRequest(
                task="Debug a distributed race condition",
                task_kind="routine_review",
                complexity="high",  # type: ignore[arg-type]
                workspace_root=str(tmp_path),
            )
        )
