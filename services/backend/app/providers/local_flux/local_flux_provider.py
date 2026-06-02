import time
from pathlib import Path

from sqlmodel import Session

from app.core.image_records import register_generated_image
from app.core.session_workspace import ensure_session_workspace
from app.db.models import GeneratedImageRecord, GenerationJobRecord, ReferenceImageRecord
from app.providers.local_flux.client import LocalFluxClient, LocalFluxClientError
from app.providers.local_flux.workflow import (
    LocalFluxWorkflowError,
    extract_history_images,
    load_workflow,
    patch_prompt,
    resolve_effective_seed,
    workflow_path_for_payload,
    workflow_to_api_prompt,
)
from app.schemas.errors import StructuredError
from app.schemas.generation import GenerationConfirmRequest
from app.settings import Settings


class LocalFluxProviderError(RuntimeError):
    def __init__(self, error: StructuredError):
        super().__init__(error.message)
        self.error = error


class LocalFluxProvider:
    def __init__(self, settings: Settings, client: LocalFluxClient | None = None):
        self.settings = settings
        self.client = client or LocalFluxClient(settings)

    def generate(
        self,
        db: Session,
        *,
        job: GenerationJobRecord,
        payload: GenerationConfirmRequest,
        reference_images: list[ReferenceImageRecord],
    ) -> list[GeneratedImageRecord]:
        try:
            self.client.get_system_stats()
        except LocalFluxClientError as error:
            raise LocalFluxProviderError(
                StructuredError(
                    code="local_flux_offline",
                    message=f"Local Flux 未連線：{self.settings.local_flux_base_url}",
                    suggestion="請先啟動本機 ComfyUI/Flux backend，或在 Local Flux 設定中修改 Server URL。",
                )
            ) from error

        try:
            uploaded_reference_names = [
                self.client.upload_image(Path(reference.storage_path))
                for reference in reference_images
            ]
            workflow_path = workflow_path_for_payload(self.settings, payload)
            workflow = load_workflow(workflow_path)
            prompt = workflow_to_api_prompt(workflow)
            effective_seed = resolve_effective_seed(payload)
            patched_prompt = patch_prompt(
                prompt,
                self.settings,
                payload,
                job_id=job.job_id,
                uploaded_reference_names=uploaded_reference_names,
                effective_seed=effective_seed,
            )
            prompt_id = self.client.post_prompt(patched_prompt, client_id=job.job_id)
            images = self._wait_for_images(prompt_id)
            return self._register_outputs(db, job, payload, images, effective_seed=effective_seed)
        except LocalFluxWorkflowError as error:
            raise LocalFluxProviderError(
                StructuredError(
                    code="local_flux_workflow_invalid",
                    message=str(error),
                    suggestion=(
                        "請在 Local Flux 設定中選擇可執行的 Flux workflow，"
                        "並確認 prompt、sampler、model、SaveImage 與 LoadImage 綁點存在。"
                    ),
                )
            ) from error
        except LocalFluxClientError as error:
            raise LocalFluxProviderError(
                StructuredError(
                    code="local_flux_generation_failed",
                    message=f"Local Flux 生成失敗：{error}",
                    suggestion="請檢查 workflow、模型路徑與本機 Flux backend log。",
                )
            ) from error

    def _wait_for_images(self, prompt_id: str) -> list[dict[str, str]]:
        deadline = time.monotonic() + self.settings.local_flux_timeout_seconds
        while time.monotonic() < deadline:
            history = self.client.get_history(prompt_id)
            images = extract_history_images(history, prompt_id)
            if images:
                return images
            time.sleep(1)
        raise LocalFluxClientError("Local Flux generation timed out before an output image appeared.")

    def _register_outputs(
        self,
        db: Session,
        job: GenerationJobRecord,
        payload: GenerationConfirmRequest,
        images: list[dict[str, str]],
        effective_seed: int,
    ) -> list[GeneratedImageRecord]:
        workspace = ensure_session_workspace(self.settings, payload.session_id)
        run_dir = workspace / "generated" / "local-flux-runs" / job.job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        records: list[GeneratedImageRecord] = []
        for index, image in enumerate(images, start=1):
            content = self.client.view_image(
                filename=image["filename"],
                subfolder=image["subfolder"],
                image_type=image["type"],
            )
            suffix = Path(image["filename"]).suffix or ".png"
            output_path = run_dir / f"local_flux_{index}{suffix}"
            output_path.write_bytes(content)
            records.append(
                register_generated_image(
                    db,
                    self.settings,
                    session_id=payload.session_id,
                    source_path=output_path,
                    provider=payload.provider,
                    seed=effective_seed,
                )
            )
        return records
