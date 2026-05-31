import re
import shutil
from pathlib import Path
from uuid import uuid4

from app.settings import Settings

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def validate_safe_id(value: str, label: str) -> None:
    if not SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{label} contains unsupported characters")


def storage_root(settings: Settings) -> Path:
    return Path(settings.storage_root).resolve()


def session_workspace(settings: Settings, session_id: str) -> Path:
    validate_safe_id(session_id, "session_id")
    return storage_root(settings) / "sessions" / session_id


def ensure_session_workspace(settings: Settings, session_id: str) -> Path:
    root = session_workspace(settings, session_id)
    for child in [
        "input",
        "generated",
        "thumbnails/reference",
        "thumbnails/generated",
    ]:
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def remove_session_workspace(settings: Settings, session_id: str) -> None:
    root = session_workspace(settings, session_id)
    if root.exists():
        shutil.rmtree(root)


def ensure_path_inside_session(settings: Settings, session_id: str, path: Path) -> Path:
    root = session_workspace(settings, session_id).resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Resolved file path is outside the session workspace")
    return resolved
