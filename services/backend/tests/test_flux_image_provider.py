from app.providers.diffusers.flux_image_provider import _device_loading_plan
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
