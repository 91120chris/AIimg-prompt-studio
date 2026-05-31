import os
import shutil
from pathlib import Path
from typing import Literal

from app.schemas.base import StrictBaseModel


ExecutableKind = Literal["native", "cmd", "ps1", "not_found", "unknown"]


class ResolvedCodexBinary(StrictBaseModel):
    configured_binary: str
    command_prefix: list[str]
    resolved_kind: ExecutableKind
    available: bool
    warning: str | None = None


def executable_kind(path: str) -> ExecutableKind:
    suffix = Path(path).suffix.lower()
    if suffix == ".cmd":
        return "cmd"
    if suffix == ".ps1":
        return "ps1"
    if suffix in {".exe", ""}:
        return "native"
    return "unknown"


def _find_windows_cmd_sibling(resolved_path: str) -> str | None:
    path = Path(resolved_path)
    cmd_sibling = path.with_suffix(".cmd")
    if cmd_sibling.exists():
        return str(cmd_sibling)
    return shutil.which(f"{path.stem}.cmd")


def _build_command_prefix(resolved_path: str, kind: ExecutableKind) -> list[str]:
    if os.name == "nt" and kind == "cmd":
        return ["cmd.exe", "/d", "/c", resolved_path]
    return [resolved_path]


def resolve_codex_binary(configured_binary: str) -> ResolvedCodexBinary:
    configured = configured_binary.strip() or "codex"
    candidate = Path(configured)

    resolved_path: str | None
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved_path = str(candidate) if candidate.exists() else None
    else:
        resolved_path = shutil.which(configured)

    if resolved_path is None:
        return ResolvedCodexBinary(
            configured_binary=configured,
            command_prefix=[],
            resolved_kind="not_found",
            available=False,
            warning="Codex CLI binary was not found on PATH.",
        )

    kind = executable_kind(resolved_path)
    warning = None

    if os.name == "nt" and kind == "ps1":
        cmd_path = _find_windows_cmd_sibling(resolved_path)
        if cmd_path is not None:
            resolved_path = cmd_path
            kind = "cmd"
        else:
            warning = (
                "Resolved Codex binary is a PowerShell script. Windows execution policy may "
                "block codex.ps1; configure CODEX_BINARY_PATH to codex.cmd or an executable path."
            )

    return ResolvedCodexBinary(
        configured_binary=configured,
        command_prefix=_build_command_prefix(resolved_path, kind),
        resolved_kind=kind,
        available=True,
        warning=warning,
    )
