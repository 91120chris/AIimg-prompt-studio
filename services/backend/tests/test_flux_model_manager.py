from app.core import flux_model_manager
from app.settings import Settings


def test_flux_defaults_target_fp8_checkpoint_repo() -> None:
    settings = Settings(_env_file=None)

    assert settings.flux_model_repo_id == "black-forest-labs/FLUX.2-klein-9b-fp8"
    assert settings.flux_model_local_dir == "local_models/huggingface/flux2-klein-9b-fp8"
    assert settings.flux_pipeline_repo_id == "black-forest-labs/FLUX.2-klein-9b"


def test_flux_install_accepts_single_file_snapshot(monkeypatch, tmp_path) -> None:
    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    checkpoint_path = downloaded_dir / "flux-2-klein-9b-fp8.safetensors"
    checkpoint_path.write_bytes(b"fake")

    def fake_snapshot_download(**_kwargs) -> str:
        return str(downloaded_dir)

    monkeypatch.setattr(flux_model_manager, "snapshot_download", fake_snapshot_download)
    settings = Settings(
        hf_token="hf_test_token",
        flux_model_local_dir=str(tmp_path / "target"),
        _env_file=None,
    )

    result = flux_model_manager.install_flux_snapshot(settings)

    assert result.model_path == str(checkpoint_path.resolve())
    assert result.repo_id == "black-forest-labs/FLUX.2-klein-9b-fp8"


def test_flux_checkpoint_path_accepts_safetensors_folder(tmp_path) -> None:
    model_dir = tmp_path / "flux-fp8"
    model_dir.mkdir()
    checkpoint_path = model_dir / "flux-2-klein-9b-fp8.safetensors"
    checkpoint_path.write_bytes(b"fake")

    assert flux_model_manager.inspect_flux_fp8_checkpoint_path(model_dir) is None
    assert flux_model_manager.select_flux_checkpoint_path(model_dir) == checkpoint_path.resolve()
