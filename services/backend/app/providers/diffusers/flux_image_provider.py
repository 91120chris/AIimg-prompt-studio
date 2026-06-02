import inspect
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.core.flux_model_manager import (
    inspect_flux_fp8_checkpoint_path,
    select_flux_checkpoint_path,
)
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
DIRECT_DEVICE_VALUES = {"cpu", "cuda", "mps", "xpu"}
CPU_OFFLOAD_VALUES = {"auto", "balanced", "balanced_low_0", "sequential"}
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
    resolved_path = Path(model_path).expanduser().resolve()
    path_problem = inspect_flux_fp8_checkpoint_path(resolved_path)
    if path_problem is not None:
        raise DiffusersFluxProviderError(
            StructuredError(
                code=path_problem.code,
                message=path_problem.message,
                suggestion=path_problem.suggestion,
            )
        )
    checkpoint_path = select_flux_checkpoint_path(resolved_path)

    try:
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from safetensors.torch import load_file
    except ImportError as error:
        raise DiffusersFluxProviderError(
            StructuredError(
                code="flux_dependencies_missing",
                message="Diffusers FLUX dependencies are not installed.",
                suggestion="Run: cd services/backend && uv sync --extra flux",
            )
        ) from error

    device_map, target_device, use_cpu_offload = _device_loading_plan(settings.flux_device_map)
    cache_key = (
        f"{checkpoint_path}|pipeline={settings.flux_pipeline_repo_id}|{settings.flux_torch_dtype}|"
        f"device_map={device_map}|device={target_device}|cpu_offload={use_cpu_offload}"
    )
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key], torch

    dtype = _torch_dtype(torch, settings.flux_torch_dtype)
    transformer_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "config": settings.flux_pipeline_repo_id,
        "subfolder": "transformer",
    }
    pipeline_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    if settings.hf_token:
        transformer_kwargs["token"] = settings.hf_token
        pipeline_kwargs["token"] = settings.hf_token
    if device_map:
        pipeline_kwargs["device_map"] = device_map

    try:
        checkpoint_state = _load_fp8_checkpoint_state_dict(load_file, checkpoint_path)
        transformer = Flux2Transformer2DModel.from_single_file(
            checkpoint_state,
            **transformer_kwargs,
        )
        del checkpoint_state
        gc.collect()
        pipe = Flux2KleinPipeline.from_pretrained(
            settings.flux_pipeline_repo_id,
            transformer=transformer,
            **pipeline_kwargs,
        )
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
    if use_cpu_offload:
        _enable_model_cpu_offload(pipe)
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe, torch


def _load_fp8_checkpoint_state_dict(load_file, checkpoint_path: Path) -> dict[str, object]:
    state_dict = load_file(str(checkpoint_path), device="cpu")
    return {
        key: value
        for key, value in state_dict.items()
        if not _is_flux_fp8_scale_metadata_key(key)
    }


def _is_flux_fp8_scale_metadata_key(key: str) -> bool:
    return key.endswith(".input_scale") or key.endswith(".weight_scale")


def _device_loading_plan(value: str | None) -> tuple[str | None, str | None, bool]:
    normalized = (value or "").strip().lower()
    if normalized in NO_DEVICE_MAP_VALUES:
        return None, None, False
    if normalized in CPU_OFFLOAD_VALUES:
        return None, None, True
    if normalized in DIRECT_DEVICE_VALUES:
        return None, normalized, False
    return normalized, None, False


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
    for method_name in ("enable_attention_slicing", "enable_vae_slicing"):
        method = getattr(pipe, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                continue


def _enable_model_cpu_offload(pipe: object) -> None:
    method = getattr(pipe, "enable_model_cpu_offload", None)
    if callable(method):
        try:
            method()
        except Exception as error:
            raise DiffusersFluxProviderError(
                StructuredError(
                    code="flux_cpu_offload_failed",
                    message="Diffusers FLUX CPU offload could not be enabled.",
                    suggestion=str(error)[:900],
                )
            ) from error


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
