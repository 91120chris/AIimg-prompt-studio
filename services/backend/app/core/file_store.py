from pathlib import Path

from fastapi import HTTPException

from app.core.session_workspace import ensure_path_inside_session
from app.db.models import GeneratedImageRecord, ReferenceImageRecord
from app.schemas.files import GeneratedImageResponse, ReferenceImageResponse
from app.settings import Settings


def reference_image_url(session_id: str, reference_image_id: str, *, thumbnail: bool = False) -> str:
    url = f"/files/sessions/{session_id}/reference-images/{reference_image_id}"
    return f"{url}?variant=thumbnail" if thumbnail else url


def generated_image_url(session_id: str, image_id: str, *, thumbnail: bool = False) -> str:
    url = f"/files/sessions/{session_id}/generated-images/{image_id}"
    return f"{url}?variant=thumbnail" if thumbnail else url


def reference_image_response(record: ReferenceImageRecord) -> ReferenceImageResponse:
    thumbnail_url = (
        reference_image_url(record.session_id, record.reference_image_id, thumbnail=True)
        if record.thumbnail_storage_path
        else None
    )
    return ReferenceImageResponse(
        reference_image_id=record.reference_image_id,
        session_id=record.session_id,
        slot=record.slot,
        role=record.role,
        url=reference_image_url(record.session_id, record.reference_image_id),
        thumbnail_url=thumbnail_url,
        filename=record.original_filename,
        width=record.width,
        height=record.height,
        created_at=record.created_at,
    )


def generated_image_response(record: GeneratedImageRecord) -> GeneratedImageResponse:
    thumbnail_url = (
        generated_image_url(record.session_id, record.image_id, thumbnail=True)
        if record.thumbnail_storage_path
        else None
    )
    return GeneratedImageResponse(
        image_id=record.image_id,
        session_id=record.session_id,
        role=record.role,
        url=generated_image_url(record.session_id, record.image_id),
        thumbnail_url=thumbnail_url,
        filename=record.filename,
        width=record.width,
        height=record.height,
        seed=record.seed,
        provider=record.provider,
        created_at=record.created_at,
    )


def select_served_path(
    settings: Settings,
    *,
    session_id: str,
    storage_path: str,
    thumbnail_storage_path: str | None,
    variant: str,
) -> Path:
    if variant not in {"original", "thumbnail"}:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_file_variant",
                "message": "File variant must be original or thumbnail.",
            },
        )

    selected = thumbnail_storage_path if variant == "thumbnail" and thumbnail_storage_path else storage_path
    try:
        path = ensure_path_inside_session(settings, session_id, Path(selected))
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "file_not_found",
                "message": "The requested file is unavailable.",
            },
        ) from None

    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "file_not_found",
                "message": "The requested file is unavailable.",
            },
        )
    return path
