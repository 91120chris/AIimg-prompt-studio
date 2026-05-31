import subprocess

from app.providers.codex.codex_binary_resolver import resolve_codex_binary
from app.schemas.provider import CodexStatusResponse
from app.settings import Settings


def get_codex_status(settings: Settings) -> CodexStatusResponse:
    resolved = resolve_codex_binary(settings.codex_binary_path)
    if not resolved.available:
        return CodexStatusResponse(
            provider="codex_cli",
            available=False,
            configured_binary=resolved.configured_binary,
            resolved_kind=resolved.resolved_kind,
            warning=resolved.warning,
            error="Codex CLI binary is unavailable.",
        )

    try:
        completed = subprocess.run(
            [*resolved.command_prefix, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except OSError as exc:
        return CodexStatusResponse(
            provider="codex_cli",
            available=False,
            configured_binary=resolved.configured_binary,
            resolved_kind=resolved.resolved_kind,
            warning=resolved.warning,
            error=str(exc),
        )
    except subprocess.TimeoutExpired:
        return CodexStatusResponse(
            provider="codex_cli",
            available=False,
            configured_binary=resolved.configured_binary,
            resolved_kind=resolved.resolved_kind,
            warning=resolved.warning,
            error="Codex CLI version check timed out.",
        )

    version = (completed.stdout or completed.stderr).strip() or None
    return CodexStatusResponse(
        provider="codex_cli",
        available=completed.returncode == 0,
        configured_binary=resolved.configured_binary,
        resolved_kind=resolved.resolved_kind,
        version=version,
        warning=resolved.warning,
        error=None if completed.returncode == 0 else "Codex CLI version check failed.",
    )
