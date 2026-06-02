import json

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.core.file_store import generated_image_response
from app.core.session_workspace import new_id
from app.db.models import GeneratedImageRecord, GenerationJobRecord, ReferenceImageRecord, SessionRecord
from app.db.session import new_session
from app.providers.codex.codex_image_provider import CodexImageProvider, CodexImageProviderError
from app.providers.local_flux.local_flux_provider import LocalFluxProvider, LocalFluxProviderError
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
                message="目前最多支援 2 張參考圖片。",
                suggestion="請移除多餘的參考圖片後再生成。",
            ),
        )

    if not reference_image_ids:
        return []

    records = db.exec(
        select(ReferenceImageRecord).where(
            ReferenceImageRecord.session_id == session_id,
            ReferenceImageRecord.reference_image_id.in_(reference_image_ids),
        ).order_by(ReferenceImageRecord.slot)
    ).all()
    found_ids = {record.reference_image_id for record in records}
    missing_ids = sorted(set(reference_image_ids) - found_ids)
    if missing_ids:
        raise _structured_http_error(
            422,
            StructuredError(
                code="reference_image_not_found",
                message=f"找不到參考圖片：{', '.join(missing_ids)}。",
                suggestion="請重新上傳參考圖片後再試一次。",
            ),
        )
    return records


@router.post("/confirm", response_model=GenerationJobResponse)
def confirm_generation(payload: GenerationConfirmRequest, request: Request) -> GenerationJobResponse:
    if payload.provider not in {"codex_cli_gpt_image", "local_flux"}:
        raise _structured_http_error(
            422,
            StructuredError(
                code="image_provider_not_implemented",
                message="目前只支援 Codex Image 與 Local Flux 生成流程。",
                suggestion="請選擇 Codex Image 或 Local Flux 後再生成。",
            ),
        )

    if not payload.optimized_prompt.strip():
        raise _structured_http_error(
            422,
            StructuredError(
                code="optimized_prompt_required",
                message="生成前需要先有最佳化 Prompt。",
                suggestion="請先完成問卷並取得最佳化 Prompt，再按下生成。",
            ),
        )

    with new_session(_engine(request)) as db:
        if db.get(SessionRecord, payload.session_id) is None:
            raise _structured_http_error(
                404,
                StructuredError(
                    code="session_not_found",
                    message="找不到工作階段。",
                    suggestion="請先建立工作階段。",
                ),
            )
        reference_records = _validate_references(db, payload.session_id, payload.reference_image_ids)

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
            if payload.provider == "local_flux":
                LocalFluxProvider(request.app.state.settings).generate(
                    db,
                    job=job,
                    payload=payload,
                    reference_images=reference_records,
                )
            else:
                CodexImageProvider(request.app.state.settings).generate(
                    db,
                    job=job,
                    payload=payload,
                    reference_images=reference_records,
                )
            job.status = "succeeded"
            job.error_json = None
        except (CodexImageProviderError, LocalFluxProviderError) as error:
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
                    suggestion="請確認 job id 後再試一次。",
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
                    suggestion="請確認 job id 後再試一次。",
                ),
            )
        return _job_response(db, job)
