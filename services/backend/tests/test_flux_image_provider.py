import pytest

from app.providers.diffusers.flux_image_provider import (
    DiffusersFluxProviderError,
    _device_loading_plan,
    _load_pipeline,
    unload_flux_pipeline,
)
from app.settings import Settings


def test_flux_default_device_map_is_balanced() -> None:
    settings = Settings(_env_file=None)

    assert settings.flux_device_map == "balanced"
    assert _device_loading_plan(settings.flux_device_map) == (None, None, True)


def test_flux_cuda_is_direct_device_not_device_map() -> None:
    assert _device_loading_plan("cuda") == (None, "cuda", False)


def test_flux_blank_device_map_disables_device_placement() -> None:
    assert _device_loading_plan("") == (None, None, False)
    assert _device_loading_plan("none") == (None, None, False)


def test_flux_provider_rejects_folder_without_checkpoint(tmp_path) -> None:
    model_dir = tmp_path / "flux-empty"
    model_dir.mkdir()

    with pytest.raises(DiffusersFluxProviderError) as exc_info:
        _load_pipeline(Settings(_env_file=None), str(model_dir))

    assert exc_info.value.error.code == "flux_checkpoint_missing"
    assert ".safetensors" in (exc_info.value.error.suggestion or "")


def test_flux_provider_uses_single_file_transformer_loader(monkeypatch, tmp_path) -> None:
    import diffusers
    import safetensors.torch

    checkpoint_path = tmp_path / "flux-2-klein-9b-fp8.safetensors"
    checkpoint_path.write_bytes(b"fake")
    calls: dict[str, object] = {}

    def fake_load_file(path: str, device: str):
        calls["load_file_path"] = path
        calls["load_file_device"] = device
        return {
            "double_blocks.0.img_attn.qkv.weight": "qkv-weight",
            "double_blocks.0.img_attn.qkv.input_scale": "input-scale",
            "double_blocks.0.img_attn.qkv.weight_scale": "weight-scale",
        }

    class FakeTransformer:
        @staticmethod
        def from_single_file(state_dict: dict[str, object], **kwargs):
            calls["checkpoint_state_dict"] = state_dict
            calls["transformer_kwargs"] = kwargs
            return "fp8-transformer"

    class FakePipe:
        def enable_model_cpu_offload(self):
            calls["cpu_offload_enabled"] = True

    class FakePipeline:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            calls["pipeline_repo_id"] = repo_id
            calls["pipeline_kwargs"] = kwargs
            return FakePipe()

    unload_flux_pipeline()
    monkeypatch.setattr(safetensors.torch, "load_file", fake_load_file)
    monkeypatch.setattr(diffusers, "Flux2Transformer2DModel", FakeTransformer)
    monkeypatch.setattr(diffusers, "Flux2KleinPipeline", FakePipeline)
    settings = Settings(
        flux_pipeline_repo_id="black-forest-labs/FLUX.2-klein-9b",
        flux_device_map="balanced",
        _env_file=None,
    )

    pipe, _torch = _load_pipeline(settings, str(checkpoint_path))

    assert isinstance(pipe, FakePipe)
    assert calls["load_file_path"] == str(checkpoint_path.resolve())
    assert calls["load_file_device"] == "cpu"
    assert calls["checkpoint_state_dict"] == {
        "double_blocks.0.img_attn.qkv.weight": "qkv-weight",
    }
    assert calls["transformer_kwargs"]["config"] == "black-forest-labs/FLUX.2-klein-9b"
    assert calls["transformer_kwargs"]["subfolder"] == "transformer"
    assert calls["pipeline_repo_id"] == "black-forest-labs/FLUX.2-klein-9b"
    assert calls["pipeline_kwargs"]["transformer"] == "fp8-transformer"
    assert "device_map" not in calls["pipeline_kwargs"]
    assert calls["cpu_offload_enabled"] is True
    unload_flux_pipeline()
