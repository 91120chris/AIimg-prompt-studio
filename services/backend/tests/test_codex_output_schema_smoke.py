import os
import subprocess

import pytest

from app.core.json_schema import schema_output_dir
from app.providers.codex.codex_binary_resolver import resolve_codex_binary
from app.providers.codex.codex_command_builder import build_codex_exec_command
from app.settings import Settings


@pytest.mark.codex_smoke
def test_codex_output_schema_smoke_is_opt_in() -> None:
    if os.environ.get("RUN_CODEX_SMOKE") != "1":
        pytest.skip("Set RUN_CODEX_SMOKE=1 to run a real Codex CLI schema smoke test.")

    settings = Settings(_env_file=None)
    binary = resolve_codex_binary(settings.codex_binary_path)
    if not binary.available:
        pytest.skip("Codex CLI is not available.")

    schema_path = schema_output_dir() / "agent_turn_response.schema.json"
    command = build_codex_exec_command(
        binary,
        "Return a valid minimal message response JSON.",
        model=settings.codex_default_model,
        output_schema_path=schema_path,
    )
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)

    assert completed.returncode == 0, completed.stderr
