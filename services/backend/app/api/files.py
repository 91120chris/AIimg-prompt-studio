from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.core.file_store import select_served_path
from app.db.models import GeneratedImageRecord, ReferenceImageRecord
from app.db.session import new_session

router = APIRouter()


def get_engine(request: Request):
    return request.app.state.engine


@router.get("/files/sessions/{session_id}/generated-images/{image_id}")
def serve_generated_image(
    session_id: str,
    image_id: str,
    request: Request,
    variant: str = "original",
) -> FileResponse:
    with new_session(get_engine(request)) as db:
        record = db.get(GeneratedImageRecord, image_id)
        if record is None or record.session_id != session_id:
            path = select_served_path(
                request.app.state.settings,
                session_id=session_id,
                storage_path="missing",
                thumbnail_storage_path=None,
                variant=variant,
            )
        else:
            path = select_served_path(
                request.app.state.settings,
                session_id=session_id,
                storage_path=record.storage_path,
                thumbnail_storage_path=record.thumbnail_storage_path,
                variant=variant,
            )
    return FileResponse(path)


@router.get("/files/sessions/{session_id}/reference-images/{reference_image_id}")
def serve_reference_image(
    session_id: str,
    reference_image_id: str,
    request: Request,
    variant: str = "original",
) -> FileResponse:
    with new_session(get_engine(request)) as db:
        record = db.get(ReferenceImageRecord, reference_image_id)
        if record is None or record.session_id != session_id:
            path = select_served_path(
                request.app.state.settings,
                session_id=session_id,
                storage_path="missing",
                thumbnail_storage_path=None,
                variant=variant,
            )
        else:
            path = select_served_path(
                request.app.state.settings,
                session_id=session_id,
                storage_path=record.storage_path,
                thumbnail_storage_path=record.thumbnail_storage_path,
                variant=variant,
            )
    return FileResponse(path)
