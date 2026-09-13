"""Command-line interface for direct use and adapter testing."""

import argparse
import asyncio
import json
from dataclasses import asdict

from llm_delegator.models import DelegationRequest
from llm_delegator.service import DelegationService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-delegator", description="Delegate a bounded task to an LLM provider"
    )
    parser.add_argument("task")
    parser.add_argument(
        "--task-kind",
        required=True,
        choices=(
            "summarize",
            "extract",
            "classify",
            "routine_transform",
            "draft_documentation",
            "draft_tests",
            "routine_code_draft",
            "routine_review",
        ),
    )
    parser.add_argument("--complexity", required=True, choices=("low", "medium"))
    parser.add_argument(
        "--accept",
        action="append",
        required=True,
        dest="acceptance_criteria",
        help="Acceptance criterion; repeat for multiple criteria",
    )
    parser.add_argument(
        "--plan-step",
        action="append",
        default=[],
        dest="plan",
        help="Ordered plan step; required for medium complexity",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        dest="constraints",
        help="Task constraint; repeat for multiple constraints",
    )
    parser.add_argument("--file", action="append", default=[], dest="files")
    parser.add_argument("--workspace-root")
    parser.add_argument("--context", default="")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument(
        "--model",
        default="auto",
        help="Model alias or ID; low-complexity tasks always use flash",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "markdown", "json", "patch"),
        default="text",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "max"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4_000)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


async def _run(args: argparse.Namespace) -> int:
    request = DelegationRequest(
        task=args.task,
        task_kind=args.task_kind,
        complexity=args.complexity,
        acceptance_criteria=tuple(args.acceptance_criteria),
        plan=tuple(args.plan),
        constraints=tuple(args.constraints),
        files=tuple(args.files),
        workspace_root=args.workspace_root,
        context=args.context,
        provider=args.provider,
        model=args.model,
        output_format=args.output_format,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
    )
    result = await DelegationService().delegate(request)
    if args.as_json:
        print(json.dumps(asdict(result), ensure_ascii=False))
    else:
        print(result.content)
    return 0


def main() -> None:
    args = _parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"llm-delegator: {error}") from error


if __name__ == "__main__":
    main()
