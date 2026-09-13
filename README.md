# LLM Delegator

LLM Delegator lets a primary agent such as GPT delegate bounded work to another
LLM through one read-only MCP tool. The primary agent remains responsible for
planning, verification, commands, and file changes.

Provider adapters support DeepSeek's Responses API and OpenRouter's Chat
Completions API without changing the MCP interface.

## Capabilities

- MCP tool: `delegate_task`
- CLI: `llm-delegator`
- Explicit, relative file selection
- Configurable allowlist of workspace roots
- Per-file and aggregate input limits
- Text, Markdown, JSON, and unified-diff output modes
- Token usage returned with every successful result
- No filesystem writes or command execution

## Delegation policy

The MCP tool accepts only explicit `low` or `medium` complexity classifications.
GPT should delegate only routine work that is bounded and easy to verify:

- Summarization, extraction, and classification
- Mechanical content or data transformations
- First drafts of documentation or tests
- Routine code drafts and routine reviews

GPT retains architecture and system design, complex debugging, ambiguous
requirements, security/privacy/auth decisions, final verification, live-state
research, external actions, and destructive work. If the primary model is
uncertain whether a task is routine, it should keep the task.

Model routing is automatic:

- `low` complexity always uses the selected provider's Flash alias.
- `medium` complexity uses the selected provider's Pro alias by default.
- OpenRouter maps those aliases to `z-ai/glm-5.3-flash` and `z-ai/glm-5.3`.
- High-complexity work cannot be submitted through the MCP schema.

Every handoff must be self-contained. GPT supplies:

- A detailed task describing the intended result
- One or more explicit acceptance criteria
- Relevant constraints
- An ordered plan whenever it helps the worker; medium-complexity tasks cannot
  run without one

This keeps the worker from guessing missing requirements or reconstructing the
primary model's reasoning.

## Install

```console
git clone https://github.com/shangzhihao/llm-delegator.git
cd llm-delegator
uv sync
```

The DeepSeek adapter reads `DEEPSEEK_API_KEY`; the OpenRouter adapter reads
`OPENROUTER_API_KEY`.

## Model configuration

Edit `llm-delegator.toml` to control which models are active:

```toml
[models]
active = [
    { provider = "deepseek", model = "deepseek-v4-flash" },
    { provider = "deepseek", model = "deepseek-v4-pro" },
    { provider = "openrouter", model = "z-ai/glm-5.3-flash" },
    { provider = "openrouter", model = "z-ai/glm-5.3" },
]
```

Each active model is identified by both provider and provider-specific model
ID. This keeps, for example, an OpenRouter-hosted GLM distinct from a future
direct Z.ai-hosted GLM with the same model ID. Removing an entry blocks that
provider/model pair before an API request. Changes take effect on the next
delegation. Set `LLM_DELEGATOR_CONFIG` to use a file at another path. If the
default `llm-delegator.toml` is absent, models are unrestricted for backward
compatibility.

## Configure Codex

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.llm-delegator]
command = "/absolute/path/to/llm-delegator/.venv/bin/llm-delegator-mcp"
cwd = "/absolute/path/to/llm-delegator"
env_vars = ["DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"]
enabled_tools = ["delegate_task"]
tool_timeout_sec = 240

[mcp_servers.llm-delegator.env]
LLM_DELEGATOR_ALLOWED_ROOTS = "/absolute/path/to/allowed/workspaces"
```

Restart Codex and use `/mcp` to confirm that `llm-delegator` is connected.
Codex can then call `delegate_task` while GPT remains the primary model.

The MCP server advertises these operating instructions to the primary model:

- Delegate only bounded, routine, easy-to-verify work.
- Write a detailed, self-contained task with acceptance criteria and constraints.
- Supply an ordered plan for medium-complexity work and whenever it would help.
- Select only the files needed for the task.
- Treat the result as an untrusted draft and verify it.
- Keep all edits and command execution with the primary model.

## CLI

The CLI exercises the same service and provider adapter as MCP:

```console
LLM_DELEGATOR_ALLOWED_ROOTS=/path/to/allowed/workspaces \
  uv run llm-delegator \
  "Summarize the provider interface" \
  --task-kind summarize \
  --complexity low \
  --accept "Describe the interface contract and method signature" \
  --constraint "Do not propose implementation changes" \
  --workspace-root /path/to/llm-delegator \
  --file src/llm_delegator/providers/base.py \
  --json
```

The CLI defaults to automatic model routing. It also accepts a provider model
alias or full model ID for direct adapter testing, but low-complexity work is
always forced to Flash. Pass `--provider openrouter` to use GLM 5.3 or GLM 5.3
Flash through OpenRouter. These models require reasoning and accept only `low`,
`high`, or `max`; the adapter maps `none` to `low` and `medium` to `high`.
Reasoning effort defaults to `high` for MCP, CLI, and programmatic requests.

## Configuration

| Environment variable | Default |
| --- | --- |
| `LLM_DELEGATOR_ALLOWED_ROOTS` | MCP process working directory |
| `LLM_DELEGATOR_CONFIG` | `./llm-delegator.toml` |
| `LLM_DELEGATOR_MAX_FILE_BYTES` | `1000000` |
| `LLM_DELEGATOR_MAX_TOTAL_FILE_BYTES` | `4000000` |
| `LLM_DELEGATOR_MAX_CONTEXT_CHARS` | `100000` |
| `LLM_DELEGATOR_REQUEST_TIMEOUT_SECONDS` | `180` |
| `LLM_DELEGATOR_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `LLM_DELEGATOR_DEEPSEEK_MODEL_FLASH` | `deepseek-v4-flash` |
| `LLM_DELEGATOR_DEEPSEEK_MODEL_PRO` | `deepseek-v4-pro` |
| `LLM_DELEGATOR_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LLM_DELEGATOR_OPENROUTER_MODEL_FLASH` | `z-ai/glm-5.3-flash` |
| `LLM_DELEGATOR_OPENROUTER_MODEL_PRO` | `z-ai/glm-5.3` |

Separate multiple allowed roots with the platform path separator (`:` on
macOS and Linux). Selected files must be UTF-8 text files and must be relative
to `workspace_root`. Symlinks cannot escape the selected workspace.

## Development

```console
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
