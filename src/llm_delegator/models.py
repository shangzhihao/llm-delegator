"""Shared request and response types."""

from dataclasses import dataclass, field
from typing import Literal

OutputFormat = Literal["text", "markdown", "json", "patch"]
ReasoningEffort = Literal["none", "low", "medium", "high", "max"]
TaskComplexity = Literal["low", "medium"]
TaskKind = Literal[
    "summarize",
    "extract",
    "classify",
    "routine_transform",
    "draft_documentation",
    "draft_tests",
    "routine_code_draft",
    "routine_review",
]


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    task: str
    task_kind: TaskKind
    complexity: TaskComplexity
    provider: str = "deepseek"
    model: str = "auto"
    workspace_root: str | None = None
    files: tuple[str, ...] = ()
    context: str = ""
    output_format: OutputFormat = "text"
    reasoning_effort: ReasoningEffort = "low"
    max_output_tokens: int = 4_000


@dataclass(frozen=True, slots=True)
class DelegationResult:
    provider: str
    model: str
    content: str
    files_read: tuple[str, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
