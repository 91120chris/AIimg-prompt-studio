from io import BytesIO
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlmodel import select

from app.core.file_store import generated_image_response, reference_image_response
from app.core.session_workspace import ensure_session_workspace, new_id, remove_session_workspace
from app.core.thumbnails import generate_thumbnail
from app.db.models import (
    AppSettingRecord,
    GeneratedImageRecord,
    PromptRecord,
    PromptVersionRecord,
    ReferenceImageRecord,
    SessionRecord,
)
from app.db.session import new_session
from app.schemas.files import GeneratedImageResponse, ReferenceImageResponse
from app.schemas.session import (
    CurrentPromptVersionPatchRequest,
    PromptVersionResponse,
    SessionCreateRequest,
    SessionResponse,
)
from app.settings import Settings

router = APIRouter()


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request):
    return request.app.state.engine


def structured_not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": "not_found", "message": message})


def _current_prompt_version_key(session_id: str) -> str:
    return f"current_prompt_version:{session_id}"


def _prompt_version_metadata(record: PromptVersionRecord) -> dict:
    try:
        value = json.loads(record.metadata_json or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _prompt_version_response(
    record: PromptVersionRecord,
    *,
    current_prompt_version_id: str | None,
) -> PromptVersionResponse:
    return PromptVersionResponse(
        prompt_version_id=record.prompt_version_id,
        session_id=record.session_id,
        prompt_text=record.prompt_text,
        title=record.title,
        source=record.source,
        metadata=_prompt_version_metadata(record),
        is_current=record.prompt_version_id == current_prompt_version_id,
        created_at=record.created_at,
    )


def _stored_current_prompt_version_id(db, session_id: str) -> str | None:
    setting = db.get(AppSettingRecord, _current_prompt_version_key(session_id))
    if setting is not None and setting.value.strip():
        record = db.get(PromptVersionRecord, setting.value)
        if record is not None and record.session_id == session_id:
            return setting.value
    latest = db.exec(
        select(PromptVersionRecord)
        .where(PromptVersionRecord.session_id == session_id)
        .order_by(PromptVersionRecord.created_at.desc(), PromptVersionRecord.prompt_version_id.desc())
    ).first()
    return latest.prompt_version_id if latest else None


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
        for model in [ReferenceImageRecord, GeneratedImageRecord, PromptRecord, PromptVersionRecord]:
            rows = db.exec(select(model).where(model.session_id == session_id)).all()
            for row in rows:
                db.delete(row)
        setting = db.get(AppSettingRecord, _current_prompt_version_key(session_id))
        if setting is not None:
            db.delete(setting)
        db.delete(record)
        db.commit()
    remove_session_workspace(settings, session_id)
    return {"status": "deleted"}


@router.get(
    "/sessions/{session_id}/prompt-versions",
    response_model=list[PromptVersionResponse],
)
def list_prompt_versions(session_id: str, request: Request) -> list[PromptVersionResponse]:
    get_session_or_404(request, session_id)
    with new_session(get_engine(request)) as db:
        current_id = _stored_current_prompt_version_id(db, session_id)
        records = db.exec(
            select(PromptVersionRecord)
            .where(PromptVersionRecord.session_id == session_id)
            .order_by(PromptVersionRecord.created_at.desc(), PromptVersionRecord.prompt_version_id.desc())
        ).all()
        return [
            _prompt_version_response(record, current_prompt_version_id=current_id)
            for record in records
        ]


@router.get(
    "/sessions/{session_id}/prompt-versions/{prompt_version_id}",
    response_model=PromptVersionResponse,
)
def get_prompt_version(
    session_id: str,
    prompt_version_id: str,
    request: Request,
) -> PromptVersionResponse:
    get_session_or_404(request, session_id)
    with new_session(get_engine(request)) as db:
        record = db.get(PromptVersionRecord, prompt_version_id)
        if record is None or record.session_id != session_id:
            raise structured_not_found("Prompt version not found.")
        return _prompt_version_response(
            record,
            current_prompt_version_id=_stored_current_prompt_version_id(db, session_id),
        )


@router.patch(
    "/sessions/{session_id}/current-prompt-version",
    response_model=PromptVersionResponse,
)
def patch_current_prompt_version(
    session_id: str,
    payload: CurrentPromptVersionPatchRequest,
    request: Request,
) -> PromptVersionResponse:
    get_session_or_404(request, session_id)
    with new_session(get_engine(request)) as db:
        record = db.get(PromptVersionRecord, payload.prompt_version_id)
        if record is None or record.session_id != session_id:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "prompt_version_not_found",
                    "message": "Prompt version not found for this session.",
                },
            )
        db.merge(
            AppSettingRecord(
                key=_current_prompt_version_key(session_id),
                value=payload.prompt_version_id,
            )
        )
        db.commit()
        return _prompt_version_response(
            record,
            current_prompt_version_id=payload.prompt_version_id,
        )


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
