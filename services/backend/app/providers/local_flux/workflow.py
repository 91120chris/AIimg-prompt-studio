import copy
import json
import secrets
from pathlib import Path
from typing import Any, Literal

from app.schemas.generation import GenerationConfirmRequest
from app.settings import Settings

WorkflowFormat = Literal["api", "ui", "unknown"]

SKIPPED_UI_NODE_TYPES = {
    "MarkdownNote",
    "Note",
    "Image Comparer (rgthree)",
}

WIDGET_INPUTS: dict[str, list[str | None]] = {
    "VAELoader": ["vae_name"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "CLIPTextEncode": ["text"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "FluxGuidance": ["guidance"],
    "LoadImage": ["image", "upload"],
    "SaveImage": ["filename_prefix"],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
}


class LocalFluxWorkflowError(ValueError):
    def __init__(self, message: str, missing_bindings: list[str] | None = None):
        super().__init__(message)
        self.missing_bindings = missing_bindings or []


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "services" / "backend").exists() and (parent / "apps").exists():
            return parent
    return Path.cwd()


def resolve_workflow_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, repo_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def workflow_path_for_payload(settings: Settings, payload: GenerationConfirmRequest) -> Path:
    if payload.mode == "t2i":
        return resolve_workflow_path(settings.local_flux_workflow_path)
    reference_count = len(payload.reference_image_ids)
    if reference_count >= 2:
        return resolve_workflow_path(settings.local_flux_i2i_two_workflow_path)
    return resolve_workflow_path(settings.local_flux_i2i_one_workflow_path)


def load_workflow(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LocalFluxWorkflowError(f"Local Flux workflow cannot be loaded: {error}") from error
    if not isinstance(payload, dict):
        raise LocalFluxWorkflowError("Local Flux workflow JSON must be an object.")
    return payload


def detect_workflow_format(workflow: dict[str, Any]) -> WorkflowFormat:
    if _is_api_prompt(workflow):
        return "api"
    if isinstance(workflow.get("nodes"), list) and isinstance(workflow.get("links"), list):
        return "ui"
    return "unknown"


def workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    workflow_format = detect_workflow_format(workflow)
    if workflow_format == "api":
        return copy.deepcopy(workflow)
    if workflow_format != "ui":
        raise LocalFluxWorkflowError("Local Flux workflow must be API-format or ComfyUI UI-format.")
    return _ui_workflow_to_api_prompt(workflow)


def patch_prompt(
    prompt: dict[str, Any],
    settings: Settings,
    payload: GenerationConfirmRequest,
    *,
    job_id: str,
    uploaded_reference_names: list[str],
    effective_seed: int | None = None,
) -> dict[str, Any]:
    patched = copy.deepcopy(prompt)
    missing: list[str] = []

    if not payload.optimized_prompt.strip():
        missing.append("positive_prompt")
    else:
        _patch_first_input(patched, ["CLIPTextEncode"], "text", payload.optimized_prompt, missing)

    _patch_first_input(
        patched,
        ["UNETLoader"],
        "unet_name",
        _resolve_unet_path(settings.local_flux_model_path),
        missing,
    )
    _patch_first_input(
        patched,
        ["VAELoader"],
        "vae_name",
        _fix_legacy_model_path(settings.local_flux_vae_path),
        missing,
    )
    _patch_first_input(
        patched,
        ["CLIPLoader"],
        "clip_name",
        _fix_legacy_model_path(settings.local_flux_text_encoder_path),
        missing,
    )

    seed = effective_seed if effective_seed is not None else resolve_effective_seed(payload)
    for key, value in {
        "seed": seed,
        "steps": settings.local_flux_steps,
        "cfg": settings.local_flux_cfg,
        "sampler_name": settings.local_flux_sampler_name,
        "scheduler": settings.local_flux_scheduler,
        "denoise": settings.local_flux_denoise,
    }.items():
        _patch_first_input(patched, ["KSampler"], key, value, missing)

    for key, value in {
        "width": settings.local_flux_width,
        "height": settings.local_flux_height,
        "batch_size": 1,
    }.items():
        _patch_first_input(
            patched,
            ["EmptyFlux2LatentImage", "EmptyLatentImage"],
            key,
            value,
            missing,
        )

    _patch_optional_first_input(patched, ["FluxGuidance"], "guidance", settings.local_flux_guidance)
    _patch_first_input(
        patched,
        ["SaveImage"],
        "filename_prefix",
        f"{settings.local_flux_output_prefix}_{job_id}",
        missing,
    )

    if payload.mode == "i2i":
        load_image_nodes = _nodes_by_class(patched, ["LoadImage"])
        if len(load_image_nodes) < len(uploaded_reference_names):
            missing.append("load_image_nodes")
        for node, image_name in zip(load_image_nodes, uploaded_reference_names, strict=False):
            node.setdefault("inputs", {})["image"] = image_name
            node.setdefault("inputs", {})["upload"] = "image"

    if missing:
        raise LocalFluxWorkflowError(
            "Local Flux workflow is missing required bindings.",
            sorted(set(missing)),
        )

    lora_name = payload.parameters.lora_name
    if lora_name and lora_name.strip():
        patched = _inject_lora_node(patched, lora_name.strip(), float(payload.parameters.lora_weight))

    return patched


def resolve_effective_seed(payload: GenerationConfirmRequest) -> int:
    if payload.parameters.seed is not None:
        return payload.parameters.seed
    return secrets.randbelow(2**32)


def validate_workflow_for_settings(
    settings: Settings,
    *,
    workflow_path: str | None,
    mode: Literal["t2i", "i2i"] = "t2i",
    reference_count: int = 0,
) -> tuple[bool, Path, WorkflowFormat, list[str], str]:
    configured_path = workflow_path or (
        settings.local_flux_workflow_path
        if mode == "t2i"
        else (
            settings.local_flux_i2i_two_workflow_path
            if reference_count >= 2
            else settings.local_flux_i2i_one_workflow_path
        )
    )
    path = resolve_workflow_path(configured_path)
    if not path.exists():
        return False, path, "unknown", ["workflow_path"], "找不到 Local Flux workflow 檔案。"
    try:
        workflow = load_workflow(path)
        workflow_format = detect_workflow_format(workflow)
        prompt = workflow_to_api_prompt(workflow)
        dummy_payload = GenerationConfirmRequest(
            session_id="session_validate",
            provider="local_flux",
            mode=mode,
            original_prompt="validate",
            optimized_prompt="validate",
            parameters={"steps": settings.local_flux_steps, "guidance": settings.local_flux_cfg},
            reference_image_ids=[f"ref_{index}" for index in range(reference_count)],
        )
        patch_prompt(
            prompt,
            settings,
            dummy_payload,
            job_id="job_validate",
            uploaded_reference_names=[f"ref_{index}.png" for index in range(reference_count)],
            effective_seed=1,
        )
    except LocalFluxWorkflowError as error:
        return False, path, detect_workflow_format(workflow) if "workflow" in locals() else "unknown", error.missing_bindings, str(error)
    return True, path, workflow_format, [], "Local Flux workflow 可用。"


def extract_history_images(history_payload: dict[str, Any], prompt_id: str) -> list[dict[str, str]]:
    entry = history_payload.get(prompt_id)
    if not isinstance(entry, dict):
        return []
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return []
    images: list[dict[str, str]] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        output_images = output.get("images")
        if not isinstance(output_images, list):
            continue
        for image in output_images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            images.append(
                {
                    "filename": filename,
                    "subfolder": str(image.get("subfolder") or ""),
                    "type": str(image.get("type") or "output"),
                }
            )
    return images


def _is_api_prompt(workflow: dict[str, Any]) -> bool:
    if not workflow:
        return False
    return all(
        isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), dict)
        for node in workflow.values()
    )


def _ui_workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise LocalFluxWorkflowError("ComfyUI UI workflow is missing nodes or links.")

    link_map: dict[int, list[Any]] = {}
    for link in links:
        if isinstance(link, list) and len(link) >= 5:
            link_map[int(link[0])] = [str(link[1]), int(link[2])]

    prompt: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        node_id = node.get("id")
        if not isinstance(node_type, str) or node_type in SKIPPED_UI_NODE_TYPES:
            continue
        if not isinstance(node_id, int | str):
            continue
        inputs: dict[str, Any] = {}
        for node_input in node.get("inputs", []):
            if not isinstance(node_input, dict):
                continue
            input_name = node_input.get("name")
            link_id = node_input.get("link")
            if isinstance(input_name, str) and isinstance(link_id, int) and link_id in link_map:
                inputs[input_name] = link_map[link_id]
        _add_widget_inputs(inputs, node_type, node.get("widgets_values", []))
        prompt[str(node_id)] = {"class_type": node_type, "inputs": inputs}

    if not prompt:
        raise LocalFluxWorkflowError("ComfyUI UI workflow did not contain executable nodes.")
    return prompt


def _add_widget_inputs(inputs: dict[str, Any], node_type: str, widgets: object) -> None:
    if not isinstance(widgets, list):
        return
    names = WIDGET_INPUTS.get(node_type)
    if names is None:
        return
    for index, input_name in enumerate(names):
        if input_name is None or index >= len(widgets) or input_name in inputs:
            continue
        inputs[input_name] = widgets[index]


def _nodes_by_class(prompt: dict[str, Any], class_types: list[str]) -> list[dict[str, Any]]:
    class_type_set = set(class_types)
    return [
        node
        for node in prompt.values()
        if isinstance(node, dict) and node.get("class_type") in class_type_set
    ]


def _patch_first_input(
    prompt: dict[str, Any],
    class_types: list[str],
    input_name: str,
    value: Any,
    missing: list[str],
) -> None:
    nodes = _nodes_by_class(prompt, class_types)
    if not nodes:
        missing.append(input_name)
        return
    nodes[0].setdefault("inputs", {})[input_name] = value


def _patch_optional_first_input(
    prompt: dict[str, Any],
    class_types: list[str],
    input_name: str,
    value: Any,
) -> None:
    nodes = _nodes_by_class(prompt, class_types)
    if not nodes:
        return
    nodes[0].setdefault("inputs", {})[input_name] = value


def _fix_legacy_model_path(path: str) -> str:
    """Strip incorrect subdirectory prefixes added by the legacy normalize function."""
    # "flux\flux2-vae.safetensors" -> "flux2-vae.safetensors" (NOT flux2\...)
    if path.startswith(("flux\\", "flux/")):
        stripped = path[5:]
        if not stripped.startswith(("flux2\\", "flux2/")):
            return stripped
    # "qwen\qwen_3_8b_fp8mixed.safetensors" -> "qwen_3_8b_fp8mixed.safetensors"
    if path.startswith(("qwen\\", "qwen/")):
        return path[5:]
    # Capitalized workaround (e.g. "Qwen_..." -> "qwen_...")
    if len(path) > 1 and path[0].isupper():
        lowered = path[0].lower() + path[1:]
        if lowered.startswith("qwen_"):
            return lowered
    return path


def _resolve_unet_path(path: str) -> str:
    """Normalize UNETLoader path to use backslash; recover from corruption."""
    fixed = path.replace("/", "\\")
    # Detect corruption (form-feed char 0x0C replacing \f in flux2\flux-...)
    if "\x0c" in fixed or (fixed and not fixed.startswith("flux2\\")):
        if "klein" in fixed or "flux-2" in fixed.lower():
            return r"flux2\flux-2-klein-9b-fp8mixed.safetensors"
    return fixed


def _inject_lora_node(prompt: dict[str, Any], lora_name: str, lora_weight: float) -> dict[str, Any]:
    """Dynamically insert a LoraLoader between the model/clip loaders and their consumers."""
    unet_id: str | None = None
    clip_id: str | None = None
    for node_id, node in prompt.items():
        ct = node.get("class_type")
        if ct == "UNETLoader" and unet_id is None:
            unet_id = node_id
        elif ct in ("CLIPLoader", "DualCLIPLoader") and clip_id is None:
            clip_id = node_id

    if not unet_id and not clip_id:
        return prompt

    lora_id = "lora_injected"
    lora_inputs: dict[str, Any] = {
        "lora_name": lora_name,
        "strength_model": lora_weight,
        "strength_clip": lora_weight,
    }
    if unet_id:
        lora_inputs["model"] = [unet_id, 0]
    if clip_id:
        lora_inputs["clip"] = [clip_id, 0]

    prompt[lora_id] = {"class_type": "LoraLoader", "inputs": lora_inputs}

    for node_id, node in prompt.items():
        if node_id == lora_id:
            continue
        for input_key, input_val in list(node.get("inputs", {}).items()):
            if not isinstance(input_val, list) or len(input_val) != 2:
                continue
            src_id, src_slot = input_val
            if unet_id and str(src_id) == str(unet_id) and src_slot == 0:
                node["inputs"][input_key] = [lora_id, 0]
            elif clip_id and str(src_id) == str(clip_id) and src_slot == 0:
                node["inputs"][input_key] = [lora_id, 1]

    return prompt
