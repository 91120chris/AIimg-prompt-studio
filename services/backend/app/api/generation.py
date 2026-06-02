import json

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.file_store import generated_image_response
from app.core.session_workspace import new_id
from app.db.models import (
    GeneratedImageRecord,
    GenerationJobRecord,
    ModelStatusRecord,
    ReferenceImageRecord,
    SessionRecord,
)
from app.db.session import new_session
from app.providers.codex.codex_image_provider import CodexImageProvider, CodexImageProviderError
from app.providers.diffusers.flux_image_provider import (
    DiffusersFluxProvider,
    DiffusersFluxProviderError,
)
from app.schemas.errors import StructuredError
from app.schemas.generation import (
    GenerationCancelRequest,
    GenerationConfirmRequest,
    GenerationImage,
    GenerationJobResponse,
)

router = APIRouter(prefix="/generation", tags=["generation"])


def _engine(request: Request):
    return request.app.state.engine


def _structured_http_error(status_code: int, error: StructuredError) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error.model_dump())


def _error_from_json(value: str | None) -> StructuredError | None:
    if not value:
        return None
    return StructuredError.model_validate(json.loads(value))


def _job_response(db, record: GenerationJobRecord) -> GenerationJobResponse:
    image_records = db.exec(
        select(GeneratedImageRecord)
        .where(GeneratedImageRecord.session_id == record.session_id)
        .order_by(GeneratedImageRecord.created_at)
    ).all()
    return GenerationJobResponse(
        job_id=record.job_id,
        session_id=record.session_id,
        provider=record.provider,
        mode=record.mode,
        status=record.status,
        images=[
            GenerationImage.model_validate(generated_image_response(image).model_dump())
            for image in image_records
        ],
        error=_error_from_json(record.error_json),
        created_at=record.created_at,
    )


def _validate_references(db, session_id: str, reference_image_ids: list[str]) -> list[ReferenceImageRecord]:
    if len(reference_image_ids) > 2:
        raise _structured_http_error(
            422,
            StructuredError(
                code="too_many_reference_images",
                message="最多只能使用 2 張參考圖片。",
                suggestion="請移除多餘的參考圖片後再試一次。",
            ),
        )

    if not reference_image_ids:
        return []

    records = db.exec(
        select(ReferenceImageRecord).where(
            ReferenceImageRecord.session_id == session_id,
            ReferenceImageRecord.reference_image_id.in_(reference_image_ids),
        )
    ).all()
    found_ids = {record.reference_image_id for record in records}
    missing_ids = sorted(set(reference_image_ids) - found_ids)
    if missing_ids:
        raise _structured_http_error(
            422,
            StructuredError(
                code="reference_image_not_found",
                message=f"找不到參考圖片：{', '.join(missing_ids)}。",
                suggestion="請確認參考圖片屬於目前 session。",
            ),
        )
    return records


def _flux_model_path(db) -> str:
    record = db.get(ModelStatusRecord, "diffusers_flux2_klein_9b_fp8")
    if record is None:
        raise _structured_http_error(
            422,
            StructuredError(
                code="flux_model_not_configured",
                message="FLUX model is not installed or configured.",
                suggestion="Use Manager to install FLUX or select a local FLUX model folder.",
            ),
        )

    try:
        details = json.loads(record.details_json)
    except json.JSONDecodeError:
        details = {}
    model_path = details.get("model_path") if isinstance(details, dict) else None
    if not isinstance(model_path, str) or not model_path.strip():
        raise _structured_http_error(
            422,
            StructuredError(
                code="flux_model_not_configured",
                message="FLUX model path is not configured.",
                suggestion="Use Manager to install FLUX or select a local FLUX model folder.",
            ),
        )
    return model_path


@router.post("/confirm", response_model=GenerationJobResponse)
def confirm_generation(payload: GenerationConfirmRequest, request: Request) -> GenerationJobResponse:
    if not payload.optimized_prompt.strip():
        raise _structured_http_error(
            422,
            StructuredError(
                code="optimized_prompt_required",
                message="生成前需要最佳化 Prompt。",
                suggestion="請先完成問卷或手動填入最佳化 Prompt。",
            ),
        )

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到目前工作階段。",
                    suggestion="請建立新的工作階段後再試一次。",
                ),
            )
        reference_records = _validate_references(db, payload.session_id, payload.reference_image_ids)
        flux_model_path = _flux_model_path(db) if payload.provider == "diffusers_flux2" else None

        job = GenerationJobRecord(
            job_id=new_id("job"),
            session_id=payload.session_id,
            provider=payload.provider,
            mode=payload.mode,
            status="running",
            parameters_json=payload.model_dump_json(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        try:
            if payload.provider == "codex_cli_gpt_image":
                CodexImageProvider(request.app.state.settings).generate(
                    db,
                    job=job,
                    payload=payload,
                    reference_images=reference_records,
                )
            elif payload.provider == "diffusers_flux2":
                DiffusersFluxProvider(request.app.state.settings).generate(
                    db,
                    job=job,
                    payload=payload,
                    model_path=flux_model_path or "",
                )
            else:
                raise _structured_http_error(
                    422,
                    StructuredError(
                        code="image_provider_not_supported",
                        message="Selected image provider is not supported.",
                        suggestion="Choose Codex GPT Image or Diffusers FLUX.",
                    ),
                )
            job.status = "succeeded"
            job.error_json = None
        except (CodexImageProviderError, DiffusersFluxProviderError) as error:
            job.status = "failed"
            job.error_json = error.error.model_dump_json()
        db.add(job)
        db.commit()
        db.refresh(job)
        return _job_response(db, job)


@router.post("/cancel", response_model=GenerationJobResponse)
def cancel_generation(payload: GenerationCancelRequest, request: Request) -> GenerationJobResponse:
    with new_session(_engine(request)) as db:
        job = db.get(GenerationJobRecord, payload.job_id)
        if job is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="generation_job_not_found",
                    message="找不到生成工作。",
                    suggestion="請確認 job id 是否正確。",
                ),
            )
        if job.status in {"succeeded", "failed", "cancelled"}:
            return _job_response(db, job)
        job.status = "cancelled"
        db.add(job)
        db.commit()
        db.refresh(job)
        return _job_response(db, job)


@router.get("/{job_id}", response_model=GenerationJobResponse)
def get_generation_job(job_id: str, request: Request) -> GenerationJobResponse:
    with new_session(_engine(request)) as db:
        job = db.get(GenerationJobRecord, job_id)
        if job is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="generation_job_not_found",
                    message="找不到生成工作。",
                    suggestion="請確認 job id 是否正確。",
                ),
            )
        return _job_response(db, job)

