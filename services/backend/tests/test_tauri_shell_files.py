from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_tauri_shell_required_files_exist() -> None:
    required = [
        "apps/desktop/src-tauri/Cargo.toml",
        "apps/desktop/src-tauri/build.rs",
        "apps/desktop/src-tauri/capabilities/default.json",
        "apps/desktop/src-tauri/icons/icon.ico",
        "apps/desktop/src-tauri/tauri.conf.json",
        "apps/desktop/src-tauri/src/lib.rs",
        "apps/desktop/src-tauri/src/main.rs",
    ]

    for relative_path in required:
        path = ROOT / relative_path
        assert path.exists(), f"{relative_path} is required for the Tauri shell"
        if path.suffix == ".ico":
            assert path.stat().st_size > 0
        else:
            assert path.read_text(encoding="utf-8").strip()


def test_tauri_dialog_plugin_is_configured() -> None:
    cargo_toml = (ROOT / "apps/desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    lib_rs = (ROOT / "apps/desktop/src-tauri/src/lib.rs").read_text(encoding="utf-8")
    main_rs = (ROOT / "apps/desktop/src-tauri/src/main.rs").read_text(encoding="utf-8")
    capability = (ROOT / "apps/desktop/src-tauri/capabilities/default.json").read_text(
        encoding="utf-8"
    )
    icon = ROOT / "apps/desktop/src-tauri/icons/icon.ico"

    assert "tauri-plugin-dialog" in cargo_toml
    assert "tauri_plugin_dialog::init()" in lib_rs
    assert "prompt_optimizer_studio_lib::run()" in main_rs
    assert "dialog:allow-open" in capability
    assert icon.read_bytes().startswith(b"\x00\x00\x01\x00")


def test_tauri_windows_icon_is_configured() -> None:
    tauri_config = (ROOT / "apps/desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8")

    assert '"icons/icon.ico"' in tauri_config
