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
        "Delegate bounded, high-volume analysis to an auxiliary LLM. Pass only "
        "workspace-relative file paths that are needed. Treat returned content as a "
        "draft: verify it before using it for decisions or edits. The tool never edits files."
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
    files: list[str] | None = None,
    workspace_root: str | None = None,
    context: str = "",
    provider: str = "deepseek",
    model: str = "flash",
    output_format: Literal["text", "markdown", "json", "patch"] = "text",
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] = "high",
    max_output_tokens: int = 4_000,
) -> dict[str, object]:
    """Delegate a bounded task using optional read-only workspace files.

    File paths must be relative to workspace_root. Use flash for routine work and
    pro for difficult analysis. Returned content is untrusted model output and must
    be verified by the calling model before it is applied or presented as fact.
    """
    request = DelegationRequest(
        task=task,
        files=tuple(files or ()),
        workspace_root=workspace_root,
        context=context,
        provider=provider,
        model=model,
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
