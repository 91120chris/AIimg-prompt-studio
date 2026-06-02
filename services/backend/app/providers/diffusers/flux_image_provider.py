import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.core.image_records import register_generated_image
from app.core.session_workspace import ensure_session_workspace
from app.db.models import GeneratedImageRecord, GenerationJobRecord
from app.schemas.errors import StructuredError
from app.schemas.generation import GenerationConfirmRequest
from app.settings import Settings


class DiffusersFluxProviderError(RuntimeError):
    def __init__(self, error: StructuredError) -> None:
        self.error = error
        super().__init__(error.message)


@dataclass
class DiffusersFluxProvider:
    settings: Settings

    def generate(
        self,
        db: Session,
        *,
        job: GenerationJobRecord,
        payload: GenerationConfirmRequest,
        model_path: str,
    ) -> list[GeneratedImageRecord]:
        if payload.mode != "t2i":
            raise DiffusersFluxProviderError(
                StructuredError(
                    code="flux_i2i_not_implemented",
                    message="Diffusers FLUX I2I is scheduled for Phase 2C.",
                    suggestion="Switch to T2I or use Codex GPT Image for reference-image generation.",
                )
            )

        pipe, torch = _load_pipeline(self.settings, model_path)
        run_dir = ensure_session_workspace(self.settings, payload.session_id) / "generated" / "flux-runs" / job.job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            output = pipe(**_pipeline_kwargs(pipe, torch, payload))
            image = output.images[0]
            output_path = run_dir / "output.png"
            image.save(output_path)
        except RuntimeError as error:
            raise _runtime_provider_error(error) from error
        except Exception as error:
            raise DiffusersFluxProviderError(
                StructuredError(
                    code="flux_generation_failed",
                    message="Diffusers FLUX generation failed.",
                    suggestion=str(error)[:900],
                )
            ) from error

        return [
            register_generated_image(
                db,
                self.settings,
                session_id=payload.session_id,
                source_path=output_path,
                provider=payload.provider,
                role="optimized_prompt",
                seed=payload.parameters.seed,
            )
        ]


_PIPELINE_CACHE: dict[str, object] = {}
DIRECT_DEVICE_VALUES = {"cuda", "mps", "xpu"}
NO_DEVICE_MAP_VALUES = {"", "none", "null", "false", "off"}


def unload_flux_pipeline() -> None:
    _PIPELINE_CACHE.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _load_pipeline(settings: Settings, model_path: str):
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as error:
        raise DiffusersFluxProviderError(
            StructuredError(
                code="flux_dependencies_missing",
                message="Diffusers FLUX dependencies are not installed.",
                suggestion="Run: cd services/backend && uv sync --extra flux",
            )
        ) from error

    resolved_path = str(Path(model_path).expanduser().resolve())
    device_map, target_device = _device_loading_plan(settings.flux_device_map)
    cache_key = (
        f"{resolved_path}|{settings.flux_torch_dtype}|"
        f"device_map={device_map}|device={target_device}"
    )
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key], torch

    dtype = _torch_dtype(torch, settings.flux_torch_dtype)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if device_map:
        kwargs["device_map"] = device_map

    try:
        pipe = DiffusionPipeline.from_pretrained(resolved_path, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        pipe = DiffusionPipeline.from_pretrained(resolved_path, **kwargs)
    except Exception as error:
        raise _load_provider_error(error) from error

    if target_device:
        try:
            moved_pipe = pipe.to(target_device)
            if moved_pipe is not None:
                pipe = moved_pipe
        except Exception as error:
            raise DiffusersFluxProviderError(
                StructuredError(
                    code="flux_device_unavailable",
                    message="The configured FLUX device could not be used.",
                    suggestion=(
                        "Set FLUX_DEVICE_MAP=balanced or FLUX_DEVICE_MAP=cpu in .env, "
                        "then restart the backend."
                    ),
                )
            ) from error

    _maybe_enable_memory_savers(pipe)
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe, torch


def _device_loading_plan(value: str | None) -> tuple[str | None, str | None]:
    normalized = (value or "").strip().lower()
    if normalized in NO_DEVICE_MAP_VALUES:
        return None, None
    if normalized in DIRECT_DEVICE_VALUES:
        return None, normalized
    return normalized, None


def _pipeline_kwargs(pipe: object, torch, payload: GenerationConfirmRequest) -> dict[str, object]:
    supports_kwargs = _supports_var_kwargs(pipe)
    call_kwargs: dict[str, object] = {}

    def add_if_supported(name: str, value: object) -> None:
        if supports_kwargs or _supports_parameter(pipe, name):
            call_kwargs[name] = value

    add_if_supported("prompt", payload.optimized_prompt)
    add_if_supported("height", payload.parameters.height)
    add_if_supported("width", payload.parameters.width)
    add_if_supported("num_inference_steps", payload.parameters.steps)
    add_if_supported("guidance_scale", payload.parameters.guidance)
    add_if_supported("output_type", "pil")
    if payload.parameters.seed is not None:
        generator = torch.Generator("cpu").manual_seed(payload.parameters.seed)
        add_if_supported("generator", generator)

    return call_kwargs


def _supports_var_kwargs(pipe: object) -> bool:
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _supports_parameter(pipe: object, name: str) -> bool:
    try:
        return name in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return True


def _torch_dtype(torch, name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _maybe_enable_memory_savers(pipe: object) -> None:
    for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_model_cpu_offload"):
        method = getattr(pipe, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                continue


def _load_provider_error(error: Exception) -> DiffusersFluxProviderError:
    message = str(error)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
        return DiffusersFluxProviderError(
            StructuredError(
                code="flux_vram_insufficient",
                message="FLUX could not be loaded with the available VRAM.",
                suggestion="Unload other GPU apps, use a smaller model, or configure CPU/offload settings.",
            )
        )
    return DiffusersFluxProviderError(
        StructuredError(
            code="flux_load_failed",
            message="Diffusers could not load the configured FLUX model.",
            suggestion=message[:900],
        )
    )


def _runtime_provider_error(error: RuntimeError) -> DiffusersFluxProviderError:
    message = str(error)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
        return DiffusersFluxProviderError(
            StructuredError(
                code="flux_vram_insufficient",
                message="FLUX generation ran out of available VRAM.",
                suggestion="Try smaller width/height, fewer steps, or unload other GPU workloads.",
            )
        )
    return DiffusersFluxProviderError(
        StructuredError(
            code="flux_runtime_error",
            message="Diffusers FLUX generation failed at runtime.",
            suggestion=message[:900],
        )
    )
