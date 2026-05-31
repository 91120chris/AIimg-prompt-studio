from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_SUFFIXES = {
    ".css",
    ".html",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
}
SKIP_PARTS = {
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    ".pytest_cache",
}
MOJIBAKE_MARKERS = [
    chr(0xFFFD),
    chr(0x00C3),
    chr(0x00C2),
    f"{chr(0x00E2)}{chr(0x20AC)}",
    f"?{chr(0x22A5)}",
    f"{chr(0x657A)}{chr(0xF5FC)}",
]


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in {"package-lock.json"}:
            continue
        files.append(path)
    return files


def test_source_files_are_clean_utf8() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for marker in MOJIBAKE_MARKERS:
            assert marker not in text, f"{path} contains suspicious text marker {marker!r}"
        for char in text:
            if char in "\n\r\t":
                continue
            assert ord(char) >= 32, f"{path} contains unexpected control character"
