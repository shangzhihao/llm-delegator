"""Environment-backed configuration."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path("llm-delegator.toml")


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


ModelIdentity = tuple[str, str]


def _active_models() -> frozenset[ModelIdentity] | None:
    configured_path = os.getenv("LLM_DELEGATOR_CONFIG")
    path = (
        Path(configured_path).expanduser() if configured_path else _DEFAULT_CONFIG_PATH
    )
    if not path.is_file():
        if configured_path:
            raise ValueError(f"LLM_DELEGATOR_CONFIG does not exist: {path}")
        return None

    try:
        with path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {path}: {error}") from error

    models = config.get("models")
    active = models.get("active") if isinstance(models, dict) else None
    if not isinstance(active, list):
        raise TypeError(
            f"{path} must define models.active as a list of provider/model tables"
        )

    identities: set[ModelIdentity] = set()
    for entry in active:
        if not isinstance(entry, dict):
            raise TypeError(
                f"{path} must define models.active as a list of provider/model tables"
            )
        provider = entry.get("provider")
        model = entry.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            raise TypeError(f"{path} model provider and model values must be strings")
        if not provider.strip() or not model.strip():
            raise ValueError(
                f"{path} model entries require non-empty provider and model strings"
            )
        identities.add((provider.strip(), model.strip()))
    return frozenset(identities)


@dataclass(frozen=True, slots=True)
class Settings:
    allowed_roots: tuple[Path, ...]
    max_file_bytes: int
    max_total_file_bytes: int
    max_context_chars: int
    request_timeout_seconds: float
    active_models: frozenset[ModelIdentity] | None = None

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
            active_models=_active_models(),
        )
