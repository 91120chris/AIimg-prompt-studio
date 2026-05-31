from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_tauri_shell_required_files_exist() -> None:
    required = [
        "apps/desktop/src-tauri/Cargo.toml",
        "apps/desktop/src-tauri/build.rs",
        "apps/desktop/src-tauri/tauri.conf.json",
        "apps/desktop/src-tauri/src/main.rs",
    ]

    for relative_path in required:
        path = ROOT / relative_path
        assert path.exists(), f"{relative_path} is required for the Tauri shell"
        assert path.read_text(encoding="utf-8").strip()
