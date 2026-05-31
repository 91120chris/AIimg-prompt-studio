from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlmodel import select

from app.core.file_store import generated_image_response, reference_image_response
from app.core.session_workspace import ensure_session_workspace, new_id, remove_session_workspace
from app.core.thumbnails import generate_thumbnail
from app.db.models import GeneratedImageRecord, ReferenceImageRecord, SessionRecord
from app.db.session import new_session
from app.schemas.files import GeneratedImageResponse, ReferenceImageResponse
from app.schemas.session import SessionCreateRequest, SessionResponse
from app.settings import Settings

router = APIRouter()


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request):
    return request.app.state.engine


def structured_not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": "not_found", "message": message})


def get_session_or_404(request: Request, session_id: str) -> SessionRecord:
    with new_session(get_engine(request)) as db:
        record = db.get(SessionRecord, session_id)
        if record is None:
            raise structured_not_found("Session not found.")
        return record


def session_response(db, record: SessionRecord) -> SessionResponse:
    reference_records = db.exec(
        select(ReferenceImageRecord)
        .where(ReferenceImageRecord.session_id == record.session_id)
        .order_by(ReferenceImageRecord.slot)
    ).all()
    generated_records = db.exec(
        select(GeneratedImageRecord)
        .where(GeneratedImageRecord.session_id == record.session_id)
        .order_by(GeneratedImageRecord.created_at)
    ).all()
    return SessionResponse(
        session_id=record.session_id,
        title=record.title,
        created_at=record.created_at,
        reference_images=[reference_image_response(item) for item in reference_records],
        generated_images=[generated_image_response(item) for item in generated_records],
    )


def validate_image_bytes(content: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            return image.width, image.height
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_image", "message": "Uploaded file is not a valid image."},
        ) from None


@router.post("/sessions", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest, request: Request) -> SessionResponse:
    settings = get_request_settings(request)
    session_id = new_id("sess")
    ensure_session_workspace(settings, session_id)
    with new_session(get_engine(request)) as db:
        record = SessionRecord(session_id=session_id, title=payload.title)
        db.add(record)
        db.commit()
        db.refresh(record)
        return session_response(db, record)


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(request: Request) -> list[SessionResponse]:
    with new_session(get_engine(request)) as db:
        records = db.exec(select(SessionRecord).order_by(SessionRecord.created_at)).all()
        return [session_response(db, record) for record in records]


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, request: Request) -> SessionResponse:
    with new_session(get_engine(request)) as db:
        record = db.get(SessionRecord, session_id)
        if record is None:
            raise structured_not_found("Session not found.")
        return session_response(db, record)


@router.delete("/sessions/{session_id}", response_model=dict[str, str])
def delete_session(session_id: str, request: Request) -> dict[str, str]:
    settings = get_request_settings(request)
    with new_session(get_engine(request)) as db:
        record = db.get(SessionRecord, session_id)
        if record is None:
            raise structured_not_found("Session not found.")
        for model in [ReferenceImageRecord, GeneratedImageRecord]:
            rows = db.exec(select(model).where(model.session_id == session_id)).all()
            for row in rows:
                db.delete(row)
        db.delete(record)
        db.commit()
    remove_session_workspace(settings, session_id)
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/reference-images", response_model=ReferenceImageResponse)
async def upload_reference_image(
    session_id: str,
    request: Request,
    slot: int = Form(...),
    role: str = Form("primary_reference"),
    file: UploadFile = File(...),
) -> ReferenceImageResponse:
    if slot not in {1, 2}:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_reference_slot", "message": "Reference image slot must be 1 or 2."},
        )

    settings = get_request_settings(request)
    content = await file.read()
    width, height = validate_image_bytes(content)
    workspace = ensure_session_workspace(settings, session_id)

    with new_session(get_engine(request)) as db:
        session_record = db.get(SessionRecord, session_id)
        if session_record is None:
            raise structured_not_found("Session not found.")

        existing = db.exec(
            select(ReferenceImageRecord).where(
                ReferenceImageRecord.session_id == session_id,
                ReferenceImageRecord.slot == slot,
            )
        ).first()
        if existing is not None:
            db.delete(existing)
            db.commit()

        reference_image_id = new_id("ref")
        original_filename = Path(file.filename or "reference.png").name
        suffix = Path(original_filename).suffix.lower() or ".png"
        filename = f"{reference_image_id}{suffix}"
        storage_path = workspace / "input" / filename
        storage_path.write_bytes(content)

        thumbnail_path = workspace / "thumbnails" / "reference" / f"{reference_image_id}.webp"
        generated_thumbnail = generate_thumbnail(storage_path, thumbnail_path)

        record = ReferenceImageRecord(
            reference_image_id=reference_image_id,
            session_id=session_id,
            slot=slot,
            role=role,
            original_filename=original_filename,
            storage_path=str(storage_path),
            thumbnail_storage_path=str(generated_thumbnail) if generated_thumbnail else None,
            width=width,
            height=height,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return reference_image_response(record)


@router.delete(
    "/sessions/{session_id}/reference-images/{slot}",
    response_model=dict[str, str],
)
def delete_reference_image(session_id: str, slot: int, request: Request) -> dict[str, str]:
    with new_session(get_engine(request)) as db:
        record = db.exec(
            select(ReferenceImageRecord).where(
                ReferenceImageRecord.session_id == session_id,
                ReferenceImageRecord.slot == slot,
            )
        ).first()
        if record is None:
            raise structured_not_found("Reference image not found.")
        db.delete(record)
        db.commit()
    return {"status": "deleted"}


@router.get(
    "/sessions/{session_id}/reference-images",
    response_model=list[ReferenceImageResponse],
)
def list_reference_images(session_id: str, request: Request) -> list[ReferenceImageResponse]:
    get_session_or_404(request, session_id)
    with new_session(get_engine(request)) as db:
        records = db.exec(
            select(ReferenceImageRecord)
            .where(ReferenceImageRecord.session_id == session_id)
            .order_by(ReferenceImageRecord.slot)
        ).all()
        return [reference_image_response(record) for record in records]


@router.get(
    "/sessions/{session_id}/generated-images",
    response_model=list[GeneratedImageResponse],
)
def list_generated_images(session_id: str, request: Request) -> list[GeneratedImageResponse]:
    get_session_or_404(request, session_id)
    with new_session(get_engine(request)) as db:
        records = db.exec(
            select(GeneratedImageRecord)
            .where(GeneratedImageRecord.session_id == session_id)
            .order_by(GeneratedImageRecord.created_at)
        ).all()
        return [generated_image_response(record) for record in records]
