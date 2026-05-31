from app.providers.codex.codex_binary_resolver import resolve_codex_binary


def test_codex_binary_resolver_reports_missing_binary() -> None:
    resolved = resolve_codex_binary("definitely-not-a-real-codex-binary")

    assert resolved.available is False
    assert resolved.resolved_kind == "not_found"
    assert resolved.command_prefix == []


def test_codex_binary_resolver_prefers_cmd_sibling_for_ps1_on_windows(tmp_path) -> None:
    ps1 = tmp_path / "codex.ps1"
    cmd = tmp_path / "codex.cmd"
    ps1.write_text("Write-Output codex", encoding="utf-8")
    cmd.write_text("@echo codex", encoding="utf-8")

    resolved = resolve_codex_binary(str(ps1))

    assert resolved.available is True
    if resolved.resolved_kind == "cmd":
        assert "codex.cmd" in resolved.command_prefix[-1]
