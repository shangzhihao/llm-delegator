"""Environment-backed configuration."""

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    allowed_roots: tuple[Path, ...]
    max_file_bytes: int
    max_total_file_bytes: int
    max_context_chars: int
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        configured = os.getenv("LLM_DELEGATOR_ALLOWED_ROOTS")
        roots = configured.split(os.pathsep) if configured else [os.getcwd()]
        resolved_roots = tuple(
            Path(root).expanduser().resolve() for root in roots if root
        )
        if not resolved_roots:
            raise ValueError("LLM_DELEGATOR_ALLOWED_ROOTS contains no usable paths")
        return cls(
            allowed_roots=resolved_roots,
            max_file_bytes=_positive_int("LLM_DELEGATOR_MAX_FILE_BYTES", 1_000_000),
            max_total_file_bytes=_positive_int(
                "LLM_DELEGATOR_MAX_TOTAL_FILE_BYTES", 4_000_000
            ),
            max_context_chars=_positive_int("LLM_DELEGATOR_MAX_CONTEXT_CHARS", 100_000),
            request_timeout_seconds=float(
                os.getenv("LLM_DELEGATOR_REQUEST_TIMEOUT_SECONDS", "180")
            ),
        )
