import pytest

from app.core import flux_model_manager
from app.settings import Settings


def test_flux_defaults_target_diffusers_pipeline_repo() -> None:
    settings = Settings(_env_file=None)

    assert settings.flux_model_repo_id == "black-forest-labs/FLUX.2-klein-9b"
    assert settings.flux_model_local_dir == "local_models/huggingface/flux2-klein-9b"


def test_flux_install_rejects_single_file_snapshot(monkeypatch, tmp_path) -> None:
    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    (downloaded_dir / "flux-2-klein-9b-fp8.safetensors").write_bytes(b"fake")

    def fake_snapshot_download(**_kwargs) -> str:
        return str(downloaded_dir)

    monkeypatch.setattr(flux_model_manager, "snapshot_download", fake_snapshot_download)
    settings = Settings(
        hf_token="hf_test_token",
        flux_model_local_dir=str(tmp_path / "target"),
        _env_file=None,
    )

    with pytest.raises(flux_model_manager.FluxInstallError) as exc_info:
        flux_model_manager.install_flux_snapshot(settings)

    assert exc_info.value.code == "flux_single_file_checkpoint_unsupported"
    assert "model_index.json" in (exc_info.value.suggestion or "")


def test_flux_pipeline_path_accepts_model_index(tmp_path) -> None:
    model_dir = tmp_path / "flux-pipeline"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    assert flux_model_manager.inspect_flux_diffusers_pipeline_path(model_dir) is None
