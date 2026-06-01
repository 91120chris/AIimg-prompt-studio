from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_tauri_shell_required_files_exist() -> None:
    required = [
        "apps/desktop/src-tauri/Cargo.toml",
        "apps/desktop/src-tauri/build.rs",
        "apps/desktop/src-tauri/capabilities/default.json",
        "apps/desktop/src-tauri/tauri.conf.json",
        "apps/desktop/src-tauri/src/main.rs",
    ]

    for relative_path in required:
        path = ROOT / relative_path
        assert path.exists(), f"{relative_path} is required for the Tauri shell"
        assert path.read_text(encoding="utf-8").strip()


def test_tauri_dialog_plugin_is_configured() -> None:
    cargo_toml = (ROOT / "apps/desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    main_rs = (ROOT / "apps/desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")
    capability = (ROOT / "apps/desktop/src-tauri/capabilities/default.json").read_text(
        encoding="utf-8"
    )

    assert "tauri-plugin-dialog" in cargo_toml
    assert "tauri_plugin_dialog::init()" in main_rs
    assert "dialog:allow-open" in capability
