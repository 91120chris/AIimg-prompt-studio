import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.flux_model_manager import (
    FluxInstallError,
    inspect_flux_fp8_checkpoint_path,
    install_flux_snapshot,
)
from app.core.session_workspace import new_id
from app.db.models import LogRecord, ModelStatusRecord
from app.db.session import new_session
from app.providers.diffusers.flux_image_provider import unload_flux_pipeline
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
    path = Path(model_path)
    if path.suffix.lower() == ".safetensors":
        return path.stem or "configured"
    return path.name or "configured"


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


def _flux_status_from_record(
    record: ModelStatusRecord | None,
    settings: Settings,
) -> FluxStatusResponse:
    default_repo_id = settings.flux_model_repo_id
    default_revision = settings.flux_model_revision
    if record is None:
        return FluxStatusResponse(
            provider=FLUX_PROVIDER,
            label=FLUX_LABEL,
            repo_id=default_repo_id,
            revision=default_revision,
            status="not_installed",
            installed=False,
            path_configured=False,
            path_label=None,
            error_code=None,
            message="FLUX provider is not installed.",
        )

    details = _details_from_record(record)
    model_path = details.get("model_path") if isinstance(details, dict) else None
    path_label = _path_label(model_path)
    repo_id = details.get("repo_id") if isinstance(details.get("repo_id"), str) else default_repo_id
    revision = (
        details.get("revision")
        if isinstance(details.get("revision"), str) or details.get("revision") is None
        else default_revision
    )
    error_code = details.get("error_code") if isinstance(details.get("error_code"), str) else None
    path_problem = inspect_flux_fp8_checkpoint_path(model_path) if isinstance(model_path, str) else None
    status = (
        record.status
        if record.status
        in {
            "not_installed",
            "path_selected",
            "install_pending",
            "installing",
            "installed",
            "loading",
            "loaded",
            "failed",
            "unloaded",
        }
        else "not_installed"
    )
    if path_problem is not None:
        status = "failed"
        error_code = path_problem.code
    installed = path_problem is None and (
        status in {"path_selected", "installed", "loading", "loaded"}
        or (status == "unloaded" and path_label is not None)
    )
    return FluxStatusResponse(
        provider=FLUX_PROVIDER,
        label=FLUX_LABEL,
        repo_id=repo_id,
        revision=revision,
        status=status,
        installed=installed,
        path_configured=path_label is not None,
        path_label=path_label,
        error_code=error_code,
        message=path_problem.message if path_problem is not None else _message_for_status(status, details),
    )


def _message_for_status(status: str, details: dict[str, object] | None = None) -> str:
    if status == "path_selected":
        return "A local FLUX model path has been configured."
    if status == "install_pending":
        return "FLUX install is queued."
    if status == "installing":
        return "FLUX model download is running."
    if status == "installed":
        return "FLUX model files are installed. T2I execution comes next in Phase 2B."
    if status == "loading":
        return "FLUX model is loading."
    if status == "loaded":
        return "FLUX model is loaded."
    if status == "failed":
        error_message = details.get("error_message") if details else None
        if isinstance(error_message, str) and error_message:
            return error_message
        return "FLUX model setup failed."
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
    path_problem = inspect_flux_fp8_checkpoint_path(model_path) if isinstance(model_path, str) else None
    hf_token_configured = settings.hf_token is not None
    hf_cache_configured = settings.hf_home is not None or settings.hf_hub_cache is not None
    can_queue_install = hf_token_configured
    if path_problem is not None:
        message = path_problem.message
    elif path_label is not None:
        message = "A local FLUX path is configured. You can install again to refresh the snapshot."
    elif can_queue_install:
        message = "HF token is configured. FLUX install can be started."
    else:
        message = "HF token is required before FLUX install can be started."
    return FluxReadinessResponse(
        provider=FLUX_PROVIDER,
        label=FLUX_LABEL,
        repo_id=settings.flux_model_repo_id,
        revision=settings.flux_model_revision,
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
    next_details = existing_details if details is None else {**existing_details, **details}
    if status != "failed":
        next_details.pop("error_code", None)
        next_details.pop("error_message", None)
    record.details_json = json.dumps(
        next_details,
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
    settings = _settings(request)
    with new_session(_engine(request)) as db:
        flux = _flux_status_from_record(_get_flux_record(db), settings)
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
                    installed=record.status in {"installed", "path_selected", "loaded"},
                    path_configured=False,
                    message=None,
                )
            )
        return items


@router.get("/flux/status", response_model=FluxStatusResponse)
def flux_status(request: Request) -> FluxStatusResponse:
    with new_session(_engine(request)) as db:
        return _flux_status_from_record(_get_flux_record(db), _settings(request))


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
                "message": "HF_TOKEN must be configured before installing FLUX.",
            },
        )

    with new_session(_engine(request)) as db:
        _upsert_flux_record(
            db,
            "installing",
            {
                "repo_id": settings.flux_model_repo_id,
                "revision": settings.flux_model_revision,
            },
            log_message="FLUX install started.",
        )

    try:
        result = install_flux_snapshot(settings)
    except FluxInstallError as error:
        with new_session(_engine(request)) as db:
            _upsert_flux_record(
                db,
                "failed",
                {
                    "repo_id": settings.flux_model_repo_id,
                    "revision": settings.flux_model_revision,
                    "error_code": error.code,
                    "error_message": error.message,
                },
                log_message=f"FLUX install failed: {error.code}",
            )
        raise HTTPException(status_code=409, detail=error.as_detail()) from error

    with new_session(_engine(request)) as db:
        path_label = _path_label(result.model_path) or "installed"
        record = _upsert_flux_record(
            db,
            "installed",
            {
                "model_path": result.model_path,
                "repo_id": result.repo_id,
                "revision": result.revision,
            },
            log_message=f"FLUX install completed: {path_label}",
        )
        return _flux_status_from_record(record, settings)


@router.post("/flux/unload", response_model=FluxStatusResponse)
def unload_flux(request: Request) -> FluxStatusResponse:
    settings = _settings(request)
    unload_flux_pipeline()
    with new_session(_engine(request)) as db:
        record = _upsert_flux_record(db, "unloaded", log_message="FLUX provider unloaded.")
        return _flux_status_from_record(record, settings)


@router.post("/flux/set-path", response_model=FluxStatusResponse)
def set_flux_path(payload: FluxPathRequest, request: Request) -> FluxStatusResponse:
    settings = _settings(request)
    model_path = payload.model_path.strip()
    if not model_path:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_flux_model_path", "message": "FLUX model path cannot be empty."},
        )
    path_problem = inspect_flux_fp8_checkpoint_path(model_path)
    if path_problem is not None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": path_problem.code,
                "message": path_problem.message,
                "suggestion": path_problem.suggestion,
            },
        )
    with new_session(_engine(request)) as db:
        path_label = _path_label(model_path) or "configured"
        record = _upsert_flux_record(
            db,
            "path_selected",
            {
                "model_path": model_path,
                "repo_id": settings.flux_model_repo_id,
                "revision": settings.flux_model_revision,
            },
            log_message=f"FLUX model path selected: {path_label}",
        )
        return _flux_status_from_record(record, settings)
