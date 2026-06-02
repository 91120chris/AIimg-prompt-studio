import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.session_workspace import new_id
from app.db.models import LogRecord, ModelStatusRecord
from app.db.session import new_session
from app.schemas.model_management import (
    FluxPathRequest,
    FluxReadinessResponse,
    FluxStatusResponse,
    ModelInfoResponse,
)
from app.settings import Settings

router = APIRouter(prefix="/models", tags=["models"])

FLUX_PROVIDER = "diffusers_flux2_klein_9b_fp8"
FLUX_LABEL = "FLUX.2 Klein 9B FP8"


def _engine(request: Request):
    return request.app.state.engine


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _path_label(model_path: object) -> str | None:
    if not isinstance(model_path, str) or not model_path.strip():
        return None
    return Path(model_path).name or "configured"


def _details_from_record(record: ModelStatusRecord | None) -> dict[str, object]:
    if record is None:
        return {}
    try:
        details = json.loads(record.details_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(details, dict):
        return {}
    return details


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

    details = _details_from_record(record)
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


def _readiness_from_record(
    record: ModelStatusRecord | None,
    settings: Settings,
) -> FluxReadinessResponse:
    details = _details_from_record(record)
    model_path = details.get("model_path")
    path_label = _path_label(model_path)
    hf_token_configured = settings.hf_token is not None
    hf_cache_configured = settings.hf_home is not None or settings.hf_hub_cache is not None
    can_queue_install = hf_token_configured
    if can_queue_install:
        message = "HF token is configured. Automatic FLUX install can be queued."
    else:
        message = "HF token is required before automatic FLUX install can be queued."
    return FluxReadinessResponse(
        provider=FLUX_PROVIDER,
        label=FLUX_LABEL,
        hf_token_configured=hf_token_configured,
        hf_cache_configured=hf_cache_configured,
        path_configured=path_label is not None,
        path_label=path_label,
        can_queue_install=can_queue_install,
        message=message,
    )


def _upsert_flux_record(
    db,
    status: str,
    details: dict[str, object] | None = None,
    log_message: str | None = None,
) -> ModelStatusRecord:
    record = db.get(ModelStatusRecord, FLUX_PROVIDER)
    existing_details = _details_from_record(record)
    if record is None:
        record = ModelStatusRecord(
            model_status_id=FLUX_PROVIDER,
            provider=FLUX_PROVIDER,
            status=status,
        )
    record.status = status
    record.details_json = json.dumps(
        existing_details if details is None else details,
        ensure_ascii=False,
        sort_keys=True,
    )
    db.add(record)
    if log_message:
        db.add(LogRecord(log_id=new_id("log"), level="info", message=log_message))
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


@router.get("/flux/readiness", response_model=FluxReadinessResponse)
def flux_readiness(request: Request) -> FluxReadinessResponse:
    with new_session(_engine(request)) as db:
        return _readiness_from_record(_get_flux_record(db), _settings(request))


@router.post("/flux/install", response_model=FluxStatusResponse)
def install_flux(request: Request) -> FluxStatusResponse:
    settings = _settings(request)
    if settings.hf_token is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "hf_token_required",
                "message": "HF_TOKEN must be configured before queuing automatic FLUX install.",
            },
        )
    with new_session(_engine(request)) as db:
        record = _upsert_flux_record(
            db,
            "install_pending",
            log_message="FLUX install marked as pending.",
        )
        return _flux_status_from_record(record)


@router.post("/flux/unload", response_model=FluxStatusResponse)
def unload_flux(request: Request) -> FluxStatusResponse:
    with new_session(_engine(request)) as db:
        record = _upsert_flux_record(db, "unloaded", log_message="FLUX provider unloaded.")
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
        path_label = _path_label(model_path) or "configured"
        record = _upsert_flux_record(
            db,
            "path_selected",
            {"model_path": model_path},
            log_message=f"FLUX model path selected: {path_label}",
        )
        return _flux_status_from_record(record)
