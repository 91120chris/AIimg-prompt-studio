import json
from io import BytesIO

from PIL import Image
from sqlmodel import select

from app.db.models import GeneratedImageRecord, GenerationJobRecord, ReferenceImageRecord
from app.db.session import new_session
from app.main import create_app
from app.providers.local_flux.local_flux_provider import LocalFluxProvider
from app.schemas.generation import GenerationConfirmRequest
from app.settings import Settings


def make_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 20), color="cyan").save(buffer, format="PNG")
    return buffer.getvalue()


def write_api_workflow(path) -> None:
    path.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "old.safetensors", "weight_dtype": "default"},
                },
                "2": {"class_type": "VAELoader", "inputs": {"vae_name": "old-vae.safetensors"}},
                "3": {
                    "class_type": "CLIPLoader",
                    "inputs": {
                        "clip_name": "old-text.safetensors",
                        "type": "flux2",
                        "device": "default",
                    },
                },
                "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "old"}},
                "5": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["4", 0],
                        "negative": ["4", 0],
                        "latent_image": ["6", 0],
                        "seed": 1,
                        "steps": 4,
                        "cfg": 1,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1,
                    },
                },
                "6": {
                    "class_type": "EmptyFlux2LatentImage",
                    "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                },
                "7": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["8", 0], "filename_prefix": "old"},
                },
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["2", 0]}},
            }
        ),
        encoding="utf-8",
    )


class FakeLocalFluxClient:
    def __init__(self):
        self.posted_prompt = None
        self.uploaded_paths = []

    def get_system_stats(self):
        return {"system": "ok"}

    def upload_image(self, path):
        self.uploaded_paths.append(path)
        return f"uploaded_{path.name}"

    def post_prompt(self, prompt, client_id):
        self.posted_prompt = prompt
        return "prompt_123"

    def get_history(self, prompt_id):
        return {
            prompt_id: {
                "outputs": {
                    "7": {
                        "images": [
                            {
                                "filename": "LocalFlux_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        }

    def view_image(self, *, filename, subfolder="", image_type="output"):
        return make_png_bytes()


def test_local_flux_provider_registers_generated_output(tmp_path) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_api_workflow(workflow_path)
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        local_flux_workflow_path=str(workflow_path),
        _env_file=None,
    )
    app = create_app(settings)
    fake_client = FakeLocalFluxClient()
    payload = GenerationConfirmRequest(
        session_id="session_local_flux",
        provider="local_flux",
        mode="t2i",
        original_prompt="城市",
        optimized_prompt="cinematic city",
        parameters={"steps": 4, "guidance": 3.5, "seed": 42},
    )

    with new_session(app.state.engine) as db:
        job = GenerationJobRecord(
            job_id="job_local_flux",
            session_id=payload.session_id,
            provider=payload.provider,
            mode=payload.mode,
            status="running",
        )
        db.add(job)
        db.commit()
        records = LocalFluxProvider(settings, client=fake_client).generate(
            db,
            job=job,
            payload=payload,
            reference_images=[],
        )
        images = db.exec(select(GeneratedImageRecord)).all()

    assert len(records) == 1
    assert len(images) == 1
    assert images[0].provider == "local_flux"
    assert images[0].seed == 42
    assert fake_client.posted_prompt["4"]["inputs"]["text"] == "cinematic city"
    assert fake_client.posted_prompt["5"]["inputs"]["seed"] == 42
    assert fake_client.posted_prompt["7"]["inputs"]["filename_prefix"] == "aiimg_job_local_flux"


def test_local_flux_provider_uses_random_effective_seed_for_blank_seed(monkeypatch, tmp_path) -> None:
    from app.providers.local_flux import workflow

    workflow_path = tmp_path / "workflow.json"
    write_api_workflow(workflow_path)
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        local_flux_workflow_path=str(workflow_path),
        _env_file=None,
    )
    app = create_app(settings)
    fake_client = FakeLocalFluxClient()
    monkeypatch.setattr(workflow.secrets, "randbelow", lambda _: 98765)
    payload = GenerationConfirmRequest(
        session_id="session_local_flux",
        provider="local_flux",
        mode="t2i",
        original_prompt="city",
        optimized_prompt="cinematic city",
        parameters={"steps": 4, "guidance": 3.5, "seed": None},
    )

    with new_session(app.state.engine) as db:
        job = GenerationJobRecord(
            job_id="job_random_seed",
            session_id=payload.session_id,
            provider=payload.provider,
            mode=payload.mode,
            status="running",
        )
        db.add(job)
        db.commit()
        LocalFluxProvider(settings, client=fake_client).generate(
            db,
            job=job,
            payload=payload,
            reference_images=[],
        )
        image = db.exec(select(GeneratedImageRecord)).one()

    assert fake_client.posted_prompt["5"]["inputs"]["seed"] == 98765
    assert image.seed == 98765


def test_local_flux_provider_uploads_and_patches_i2i_references(tmp_path) -> None:
    workflow_path = tmp_path / "i2i_workflow.json"
    write_api_workflow(workflow_path)
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    prompt["9"] = {"class_type": "LoadImage", "inputs": {"image": "old_1.png"}}
    prompt["10"] = {"class_type": "LoadImage", "inputs": {"image": "old_2.png"}}
    workflow_path.write_text(json.dumps(prompt), encoding="utf-8")
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        local_flux_i2i_two_workflow_path=str(workflow_path),
        _env_file=None,
    )
    app = create_app(settings)
    fake_client = FakeLocalFluxClient()
    ref_one = tmp_path / "ref_one.png"
    ref_two = tmp_path / "ref_two.png"
    ref_one.write_bytes(make_png_bytes())
    ref_two.write_bytes(make_png_bytes())
    payload = GenerationConfirmRequest(
        session_id="session_local_flux",
        provider="local_flux",
        mode="i2i",
        original_prompt="edit",
        optimized_prompt="edit with references",
        parameters={"steps": 4, "guidance": 3.5, "seed": 123},
        reference_image_ids=["ref_one", "ref_two"],
    )

    with new_session(app.state.engine) as db:
        job = GenerationJobRecord(
            job_id="job_i2i",
            session_id=payload.session_id,
            provider=payload.provider,
            mode=payload.mode,
            status="running",
        )
        db.add(job)
        db.commit()
        LocalFluxProvider(settings, client=fake_client).generate(
            db,
            job=job,
            payload=payload,
            reference_images=[
                ReferenceImageRecord(
                    reference_image_id="ref_one",
                    session_id=payload.session_id,
                    slot=1,
                    role="primary_reference",
                    original_filename="ref_one.png",
                    storage_path=str(ref_one),
                    width=24,
                    height=20,
                ),
                ReferenceImageRecord(
                    reference_image_id="ref_two",
                    session_id=payload.session_id,
                    slot=2,
                    role="secondary_reference",
                    original_filename="ref_two.png",
                    storage_path=str(ref_two),
                    width=24,
                    height=20,
                ),
            ],
        )

    assert [path.name for path in fake_client.uploaded_paths] == ["ref_one.png", "ref_two.png"]
    assert fake_client.posted_prompt["9"]["inputs"]["image"] == "uploaded_ref_one.png"
    assert fake_client.posted_prompt["10"]["inputs"]["image"] == "uploaded_ref_two.png"
