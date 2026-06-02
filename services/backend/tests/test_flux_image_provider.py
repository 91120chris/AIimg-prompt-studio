import pytest

from app.providers.diffusers.flux_image_provider import (
    DiffusersFluxProviderError,
    _device_loading_plan,
    _load_pipeline,
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


def test_flux_provider_rejects_single_file_checkpoint_folder(tmp_path) -> None:
    model_dir = tmp_path / "flux-single-file"
    model_dir.mkdir()
    (model_dir / "flux-2-klein-9b-fp8.safetensors").write_bytes(b"fake")

    with pytest.raises(DiffusersFluxProviderError) as exc_info:
        _load_pipeline(Settings(_env_file=None), str(model_dir))

    assert exc_info.value.error.code == "flux_single_file_checkpoint_unsupported"
    assert "model_index.json" in (exc_info.value.error.suggestion or "")
