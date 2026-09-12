"""Constrained, read-only workspace access."""

from pathlib import Path

from llm_delegator.config import Settings


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_workspace_root(requested_root: str | None, settings: Settings) -> Path:
    root = (
        Path(requested_root).expanduser().resolve()
        if requested_root
        else Path.cwd().resolve()
    )
    if not any(_is_within(root, allowed) for allowed in settings.allowed_roots):
        allowed = ", ".join(str(path) for path in settings.allowed_roots)
        raise ValueError(f"Workspace root is outside the allowed roots: {allowed}")
    if not root.is_dir():
        raise ValueError(f"Workspace root is not a directory: {root}")
    return root


def read_workspace_files(
    workspace_root: Path, files: tuple[str, ...], settings: Settings
) -> tuple[tuple[str, str], ...]:
    loaded: list[tuple[str, str]] = []
    total_bytes = 0

    for requested in files:
        relative = Path(requested)
        if relative.is_absolute():
            raise ValueError(
                f"File paths must be relative to the workspace root: {requested}"
            )

        path = (workspace_root / relative).resolve(strict=True)
        if not _is_within(path, workspace_root):
            raise ValueError(f"File resolves outside the workspace root: {requested}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {requested}")

        size = path.stat().st_size
        if size > settings.max_file_bytes:
            raise ValueError(
                f"File exceeds the {settings.max_file_bytes}-byte limit: {requested}"
            )
        total_bytes += size
        if total_bytes > settings.max_total_file_bytes:
            raise ValueError(
                "Selected files exceed the total byte limit of "
                f"{settings.max_total_file_bytes}"
            )

        data = path.read_bytes()
        if b"\x00" in data:
            raise ValueError(f"Binary files are not supported: {requested}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"File is not valid UTF-8: {requested}") from error
        loaded.append((relative.as_posix(), text))

    return tuple(loaded)
