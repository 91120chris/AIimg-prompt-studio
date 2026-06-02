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
    assert _device_loading_plan(settings.flux_device_map) == ("balanced", None)


def test_flux_cuda_is_direct_device_not_device_map() -> None:
    assert _device_loading_plan("cuda") == (None, "cuda")


def test_flux_blank_device_map_disables_device_placement() -> None:
    assert _device_loading_plan("") == (None, None)
    assert _device_loading_plan("none") == (None, None)


def test_flux_provider_rejects_folder_without_checkpoint(tmp_path) -> None:
    model_dir = tmp_path / "flux-empty"
    model_dir.mkdir()

    with pytest.raises(DiffusersFluxProviderError) as exc_info:
        _load_pipeline(Settings(_env_file=None), str(model_dir))

    assert exc_info.value.error.code == "flux_checkpoint_missing"
    assert ".safetensors" in (exc_info.value.error.suggestion or "")


def test_flux_provider_uses_single_file_transformer_loader(monkeypatch, tmp_path) -> None:
    import diffusers

    checkpoint_path = tmp_path / "flux-2-klein-9b-fp8.safetensors"
    checkpoint_path.write_bytes(b"fake")
    calls: dict[str, object] = {}

    class FakeTransformer:
        @staticmethod
        def from_single_file(path: str, **kwargs):
            calls["checkpoint_path"] = path
            calls["transformer_kwargs"] = kwargs
            return "fp8-transformer"

    class FakePipe:
        pass

    class FakePipeline:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs):
            calls["pipeline_repo_id"] = repo_id
            calls["pipeline_kwargs"] = kwargs
            return FakePipe()

    unload_flux_pipeline()
    monkeypatch.setattr(diffusers, "Flux2Transformer2DModel", FakeTransformer)
    monkeypatch.setattr(diffusers, "Flux2KleinPipeline", FakePipeline)
    settings = Settings(
        flux_pipeline_repo_id="black-forest-labs/FLUX.2-klein-9b",
        flux_device_map="",
        _env_file=None,
    )

    pipe, _torch = _load_pipeline(settings, str(checkpoint_path))

    assert isinstance(pipe, FakePipe)
    assert calls["checkpoint_path"] == str(checkpoint_path.resolve())
    assert calls["transformer_kwargs"]["config"] == "black-forest-labs/FLUX.2-klein-9b"
    assert calls["transformer_kwargs"]["subfolder"] == "transformer"
    assert calls["pipeline_repo_id"] == "black-forest-labs/FLUX.2-klein-9b"
    assert calls["pipeline_kwargs"]["transformer"] == "fp8-transformer"
    unload_flux_pipeline()
