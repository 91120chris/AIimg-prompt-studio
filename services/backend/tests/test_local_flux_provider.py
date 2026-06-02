import json
from io import BytesIO

from PIL import Image
from sqlmodel import select

from app.db.models import GeneratedImageRecord, GenerationJobRecord
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

    def get_system_stats(self):
        return {"system": "ok"}

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
