from pathlib import Path

import pytest

from llm_delegator.config import Settings
from llm_delegator.workspace import read_workspace_files, resolve_workspace_root


def settings(root: Path) -> Settings:
    return Settings(
        allowed_roots=(root.resolve(),),
        max_file_bytes=100,
        max_total_file_bytes=150,
        max_context_chars=1_000,
        request_timeout_seconds=10,
    )


def test_reads_relative_utf8_file(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("answer = 42\n", encoding="utf-8")

    root = resolve_workspace_root(str(source), settings(tmp_path))
    loaded = read_workspace_files(root, ("example.py",), settings(tmp_path))

    assert loaded == (("example.py", "answer = 42\n"),)


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the workspace root"):
        read_workspace_files(
            workspace.resolve(), ("../secret.txt",), settings(tmp_path)
        )


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="outside the workspace root"):
        read_workspace_files(workspace.resolve(), ("link.txt",), settings(tmp_path))


def test_rejects_workspace_outside_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()

    with pytest.raises(ValueError, match="outside the allowed roots"):
        resolve_workspace_root(str(denied), settings(allowed))
