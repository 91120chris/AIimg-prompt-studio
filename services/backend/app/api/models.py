import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.db.models import ModelStatusRecord
from app.db.session import new_session
from app.schemas.model_management import FluxPathRequest, FluxStatusResponse, ModelInfoResponse

router = APIRouter(prefix="/models", tags=["models"])

FLUX_PROVIDER = "diffusers_flux2_klein_9b_fp8"
FLUX_LABEL = "FLUX.2 Klein 9B FP8"


def _engine(request: Request):
    return request.app.state.engine


def _path_label(model_path: object) -> str | None:
    if not isinstance(model_path, str) or not model_path.strip():
        return None
    return Path(model_path).name or "configured"


def _flux_status_from_record(record: ModelStatusRecord | None) -> FluxStatusResponse:
    if record is None:
        return FluxStatusResponse(
            provider=FLUX_PROVIDER,
            label=FLUX_LABEL,
            status="not_installed",
            installed=False,
            path_configured=False,
            path_label=None,
            message="Phase 2 provider placeholder. Set a local path or implement installer next.",
        )

    try:
        details = json.loads(record.details_json)
    except json.JSONDecodeError:
        details = {}
    model_path = details.get("model_path") if isinstance(details, dict) else None
    path_label = _path_label(model_path)
    status = (
        record.status
        if record.status in {"not_installed", "path_selected", "install_pending", "unloaded"}
        else "not_installed"
    )
    installed = status == "path_selected"
    return FluxStatusResponse(
        provider=FLUX_PROVIDER,
        label=FLUX_LABEL,
        status=status,
        installed=installed,
        path_configured=path_label is not None,
        path_label=path_label,
        message=_message_for_status(status),
    )


def _message_for_status(status: str) -> str:
    if status == "path_selected":
        return "A local FLUX model path has been configured; generation is still Phase 2 work."
    if status == "install_pending":
        return "Automatic FLUX install is queued for Phase 2 implementation."
    if status == "unloaded":
        return "FLUX provider is unloaded."
    return "FLUX provider is not installed."


def _get_flux_record(db) -> ModelStatusRecord | None:
    return db.get(ModelStatusRecord, FLUX_PROVIDER)


def _upsert_flux_record(db, status: str, details: dict[str, object] | None = None) -> ModelStatusRecord:
    record = db.get(ModelStatusRecord, FLUX_PROVIDER)
    if record is None:
        record = ModelStatusRecord(
            model_status_id=FLUX_PROVIDER,
            provider=FLUX_PROVIDER,
            status=status,
        )
    record.status = status
    record.details_json = json.dumps(details or {}, ensure_ascii=False, sort_keys=True)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("", response_model=list[ModelInfoResponse])
def list_models(request: Request) -> list[ModelInfoResponse]:
    with new_session(_engine(request)) as db:
        flux = _flux_status_from_record(_get_flux_record(db))
        records = db.exec(select(ModelStatusRecord)).all()
        items = [
            ModelInfoResponse(
                provider=flux.provider,
                label=flux.label,
                status=flux.status,
                installed=flux.installed,
                path_configured=flux.path_configured,
                path_label=flux.path_label,
                message=flux.message,
            )
        ]
        for record in records:
            if record.provider == FLUX_PROVIDER:
                continue
            items.append(
                ModelInfoResponse(
                    provider=record.provider,
                    label=record.provider,
                    status=record.status,
                    installed=record.status in {"installed", "path_selected"},
                    path_configured=False,
                    message=None,
                )
            )
        return items


@router.get("/flux/status", response_model=FluxStatusResponse)
def flux_status(request: Request) -> FluxStatusResponse:
    with new_session(_engine(request)) as db:
        return _flux_status_from_record(_get_flux_record(db))


@router.post("/flux/install", response_model=FluxStatusResponse)
def install_flux(request: Request) -> FluxStatusResponse:
    with new_session(_engine(request)) as db:
        record = _upsert_flux_record(db, "install_pending")
        return _flux_status_from_record(record)


@router.post("/flux/unload", response_model=FluxStatusResponse)
def unload_flux(request: Request) -> FluxStatusResponse:
    with new_session(_engine(request)) as db:
        record = _upsert_flux_record(db, "unloaded")
        return _flux_status_from_record(record)


@router.post("/flux/set-path", response_model=FluxStatusResponse)
def set_flux_path(payload: FluxPathRequest, request: Request) -> FluxStatusResponse:
    model_path = payload.model_path.strip()
    if not model_path:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_flux_model_path", "message": "FLUX model path cannot be empty."},
        )
    with new_session(_engine(request)) as db:
        record = _upsert_flux_record(db, "path_selected", {"model_path": model_path})
        return _flux_status_from_record(record)
