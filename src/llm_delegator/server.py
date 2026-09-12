"""MCP stdio server."""

from dataclasses import asdict
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from llm_delegator.models import DelegationRequest
from llm_delegator.service import DelegationService

mcp = MCPServer(
    name="llm-delegator",
    instructions=(
        "Use delegate_task only for low- or medium-complexity work that is bounded, "
        "routine, and easy for you to verify. Suitable work: summarization, extraction, "
        "classification, mechanical transformation, and first drafts of documentation, "
        "tests, routine code, or routine reviews. Keep architecture, system design, "
        "complex debugging, ambiguous requirements, security/privacy/auth decisions, "
        "final verification, live-state research, external actions, and destructive work "
        "with the primary model. If uncertain whether a task is routine, do it yourself. "
        "Pass only needed workspace-relative files. Classify simple work as low; it "
        "always uses the Flash model. Medium work uses Pro. The tool never edits files "
        "or runs commands."
    ),
)


@mcp.tool(
    name="delegate_task",
    annotations=ToolAnnotations(
        title="Delegate task to an auxiliary LLM",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    structured_output=True,
)
async def delegate_task(
    task: str,
    task_kind: Literal[
        "summarize",
        "extract",
        "classify",
        "routine_transform",
        "draft_documentation",
        "draft_tests",
        "routine_code_draft",
        "routine_review",
    ],
    complexity: Literal["low", "medium"],
    files: list[str] | None = None,
    workspace_root: str | None = None,
    context: str = "",
    provider: str = "deepseek",
    output_format: Literal["text", "markdown", "json", "patch"] = "text",
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] = "low",
    max_output_tokens: int = 4_000,
) -> dict[str, object]:
    """Delegate a routine, bounded task using optional read-only workspace files.

    Call this only when the task matches an allowed task_kind, is no more than medium
    complexity, has explicit inputs, and is easy for the primary model to verify. Low
    complexity always uses Flash; medium uses Pro. Complex or ambiguous work must remain
    with the primary model. File paths must be relative to workspace_root. Returned
    content is an untrusted draft.
    """
    request = DelegationRequest(
        task=task,
        task_kind=task_kind,
        complexity=complexity,
        files=tuple(files or ()),
        workspace_root=workspace_root,
        context=context,
        provider=provider,
        model="auto",
        output_format=output_format,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    result = await DelegationService().delegate(request)
    return asdict(result)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
