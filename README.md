# LLM Delegator

LLM Delegator lets a primary agent such as GPT delegate bounded work to another
LLM through one read-only MCP tool. The primary agent remains responsible for
planning, verification, commands, and file changes.

The first provider adapter uses DeepSeek's Responses API. The provider boundary
is intentionally small so additional providers can be added without changing
the MCP interface.

## Capabilities

- MCP tool: `delegate_task`
- CLI: `llm-delegator`
- Explicit, relative file selection
- Configurable allowlist of workspace roots
- Per-file and aggregate input limits
- Text, Markdown, JSON, and unified-diff output modes
- Token usage returned with every successful result
- No filesystem writes or command execution

## Install

```console
git clone https://github.com/shangzhihao/llm-delegator.git
cd llm-delegator
uv sync
```

The DeepSeek adapter reads `DEEPSEEK_API_KEY` from its environment.

## Configure Codex

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.llm-delegator]
command = "/absolute/path/to/llm-delegator/.venv/bin/llm-delegator-mcp"
cwd = "/absolute/path/to/llm-delegator"
env_vars = ["DEEPSEEK_API_KEY"]
enabled_tools = ["delegate_task"]
tool_timeout_sec = 240

[mcp_servers.llm-delegator.env]
LLM_DELEGATOR_ALLOWED_ROOTS = "/absolute/path/to/allowed/workspaces"
```

Restart Codex and use `/mcp` to confirm that `llm-delegator` is connected.
Codex can then call `delegate_task` while GPT remains the primary model.

The MCP server advertises these operating instructions to the primary model:

- Delegate bounded, high-volume analysis.
- Select only the files needed for the task.
- Treat the result as an untrusted draft and verify it.
- Keep all edits and command execution with the primary model.

## CLI

The CLI exercises the same service and provider adapter as MCP:

```console
LLM_DELEGATOR_ALLOWED_ROOTS=/path/to/allowed/workspaces \
  uv run llm-delegator \
  "Summarize the provider interface" \
  --workspace-root /path/to/llm-delegator \
  --file src/llm_delegator/providers/base.py \
  --model flash \
  --json
```

Use `--model pro` for harder tasks. A full provider model ID can be passed in
place of an alias.

## Configuration

| Environment variable | Default |
| --- | --- |
| `LLM_DELEGATOR_ALLOWED_ROOTS` | MCP process working directory |
| `LLM_DELEGATOR_MAX_FILE_BYTES` | `1000000` |
| `LLM_DELEGATOR_MAX_TOTAL_FILE_BYTES` | `4000000` |
| `LLM_DELEGATOR_MAX_CONTEXT_CHARS` | `100000` |
| `LLM_DELEGATOR_REQUEST_TIMEOUT_SECONDS` | `180` |
| `LLM_DELEGATOR_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `LLM_DELEGATOR_DEEPSEEK_MODEL_FLASH` | `deepseek-v4-flash` |
| `LLM_DELEGATOR_DEEPSEEK_MODEL_PRO` | `deepseek-v4-pro` |

Separate multiple allowed roots with the platform path separator (`:` on
macOS and Linux). Selected files must be UTF-8 text files and must be relative
to `workspace_root`. Symlinks cannot escape the selected workspace.

## Development

```console
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
