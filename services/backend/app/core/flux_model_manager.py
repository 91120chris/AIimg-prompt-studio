from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

from app.settings import Settings

DEFAULT_FLUX_DIFFUSERS_REPO_ID = "black-forest-labs/FLUX.2-klein-9b"
DEFAULT_FLUX_DIFFUSERS_LOCAL_DIR = "local_models/huggingface/flux2-klein-9b"
DIFFUSERS_MODEL_INDEX = "model_index.json"
DIFFUSERS_PIPELINE_SUGGESTION = (
    "Install black-forest-labs/FLUX.2-klein-9b, or choose a local Diffusers pipeline "
    "folder that contains model_index.json. The FP8 FLUX.2 Klein repos are single-file "
    "checkpoints and are not supported by this provider yet."
)


@dataclass(frozen=True)
class FluxInstallResult:
    model_path: str
    repo_id: str
    revision: str | None


@dataclass(frozen=True)
class FluxModelPathProblem:
    code: str
    message: str
    suggestion: str


class FluxInstallError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion

    def as_detail(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
        }


def install_flux_snapshot(settings: Settings) -> FluxInstallResult:
    if settings.hf_token is None:
        raise FluxInstallError(
            code="hf_token_required",
            message="HF_TOKEN must be configured before installing FLUX.",
            suggestion="Set HF_TOKEN in .env, then restart the backend.",
        )

    local_dir = _resolve_local_dir(settings.flux_model_local_dir)
    cache_dir = _resolve_cache_dir(settings)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=settings.flux_model_repo_id,
            revision=settings.flux_model_revision,
            token=settings.hf_token,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except GatedRepoError as error:
        raise FluxInstallError(
            code="hf_gated_repo",
            message="FLUX model access is gated or license acceptance is missing.",
            suggestion=(
                "Log in to Hugging Face, accept the model terms, confirm HF_TOKEN has access, "
                "then retry install."
            ),
        ) from error
    except RepositoryNotFoundError as error:
        raise FluxInstallError(
            code="hf_repo_not_found_or_private",
            message="The configured FLUX repository was not found or is private.",
            suggestion="Check FLUX_MODEL_REPO_ID and make sure HF_TOKEN can access it.",
        ) from error
    except LocalEntryNotFoundError as error:
        raise FluxInstallError(
            code="hf_local_cache_missing",
            message="The requested FLUX files were not found in the local Hugging Face cache.",
            suggestion="Connect to the internet or choose a local model folder manually.",
        ) from error
    except HfHubHTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        raise _http_install_error(status_code) from error
    except OSError as error:
        raise FluxInstallError(
            code="flux_install_filesystem_error",
            message="FLUX install could not write to the configured local model directory.",
            suggestion="Choose a writable model folder and make sure enough disk space is available.",
        ) from error
    except Exception as error:
        raise FluxInstallError(
            code="flux_install_failed",
            message="FLUX install failed before the model could be registered.",
            suggestion="Check network access, Hugging Face permissions, and available disk space.",
        ) from error

    problem = inspect_flux_diffusers_pipeline_path(downloaded_path)
    if problem is not None:
        raise FluxInstallError(
            code=problem.code,
            message=problem.message,
            suggestion=problem.suggestion,
        )

    return FluxInstallResult(
        model_path=str(Path(downloaded_path).resolve()),
        repo_id=settings.flux_model_repo_id,
        revision=settings.flux_model_revision,
    )


def inspect_flux_diffusers_pipeline_path(model_path: str | Path) -> FluxModelPathProblem | None:
    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.exists():
        return FluxModelPathProblem(
            code="flux_model_path_missing",
            message="The configured FLUX model path does not exist.",
            suggestion="Install FLUX again or choose an existing local model folder.",
        )
    if resolved_path.is_file():
        if resolved_path.suffix.lower() == ".safetensors":
            return FluxModelPathProblem(
                code="flux_single_file_checkpoint_unsupported",
                message="The configured FLUX path is a single-file checkpoint, not a Diffusers pipeline folder.",
                suggestion=DIFFUSERS_PIPELINE_SUGGESTION,
            )
        return FluxModelPathProblem(
            code="flux_model_path_not_directory",
            message="The configured FLUX model path is a file, not a Diffusers pipeline folder.",
            suggestion=DIFFUSERS_PIPELINE_SUGGESTION,
        )
    if not (resolved_path / DIFFUSERS_MODEL_INDEX).is_file():
        safetensors = list(resolved_path.glob("*.safetensors"))
        if safetensors:
            return FluxModelPathProblem(
                code="flux_single_file_checkpoint_unsupported",
                message=(
                    "The configured FLUX folder contains single-file checkpoint weights, "
                    "but not a Diffusers pipeline model_index.json."
                ),
                suggestion=DIFFUSERS_PIPELINE_SUGGESTION,
            )
        return FluxModelPathProblem(
            code="flux_model_index_missing",
            message="The configured FLUX folder is missing Diffusers model_index.json.",
            suggestion=DIFFUSERS_PIPELINE_SUGGESTION,
        )
    return None


def _http_install_error(status_code: int | None) -> FluxInstallError:
    if status_code == 401:
        return FluxInstallError(
            code="hf_token_invalid",
            message="Hugging Face rejected the configured token.",
            suggestion="Check HF_TOKEN, rotate it if needed, and restart the backend.",
        )
    if status_code == 403:
        return FluxInstallError(
            code="hf_access_denied",
            message="Hugging Face denied access to the configured FLUX repository.",
            suggestion="Accept the model terms and make sure the token has permission.",
        )
    return FluxInstallError(
        code="hf_download_error",
        message="Hugging Face returned an error while downloading FLUX.",
        suggestion="Retry later or choose an already downloaded local model folder.",
    )


def _resolve_local_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return _project_root() / path


def _resolve_model_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_project_root() / path).resolve()


def _resolve_cache_dir(settings: Settings) -> Path | None:
    if settings.hf_hub_cache:
        return Path(settings.hf_hub_cache).expanduser()
    if settings.hf_home:
        return Path(settings.hf_home).expanduser() / "hub"
    return None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]
