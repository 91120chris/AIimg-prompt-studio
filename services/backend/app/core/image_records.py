import shutil
from pathlib import Path

from sqlmodel import Session

from app.core.session_workspace import ensure_session_workspace, new_id
from app.core.thumbnails import generate_thumbnail, image_dimensions
from app.db.models import GeneratedImageRecord
from app.settings import Settings


def register_generated_image(
    db: Session,
    settings: Settings,
    *,
    session_id: str,
    source_path: Path,
    provider: str,
    role: str = "optimized_prompt",
    seed: int | None = None,
) -> GeneratedImageRecord:
    workspace = ensure_session_workspace(settings, session_id)
    image_id = new_id("img")
    suffix = source_path.suffix.lower() or ".png"
    filename = f"{image_id}{suffix}"
    destination = workspace / "generated" / filename
    shutil.copyfile(source_path, destination)
    width, height = image_dimensions(destination)

    thumbnail_path = workspace / "thumbnails" / "generated" / f"{image_id}.webp"
    generated_thumbnail = generate_thumbnail(destination, thumbnail_path)

    record = GeneratedImageRecord(
        image_id=image_id,
        session_id=session_id,
        role=role,
        filename=filename,
        storage_path=str(destination),
        thumbnail_storage_path=str(generated_thumbnail) if generated_thumbnail else None,
        width=width,
        height=height,
        seed=seed,
        provider=provider,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
