import json

from PIL import Image

from app.db.models import GenerationJobRecord
from app.db.session import create_db_engine, init_db, new_session
from app.providers.codex import codex_image_provider
from app.providers.codex.codex_binary_resolver import ResolvedCodexBinary
from app.providers.codex.codex_image_provider import CodexImageProvider
from app.schemas.generation import GenerationConfirmRequest
from app.settings import Settings


def fake_binary() -> ResolvedCodexBinary:
    return ResolvedCodexBinary(
        configured_binary="codex",
        command_prefix=["codex"],
        resolved_kind="native",
        available=True,
    )


def test_codex_image_provider_registers_generated_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(codex_image_provider, "resolve_codex_binary", lambda _: fake_binary())
    settings = Settings(
        storage_root=str(tmp_path / "storage"),
        database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
        _env_file=None,
    )
    engine = create_db_engine(settings)
    init_db(engine)
    commands: list[list[str]] = []
    prompts: list[str] = []

    def fake_executor(command, timeout_seconds, input_text, cwd):
        commands.append(command)
        prompts.append(input_text)
        Image.new("RGB", (32, 24), color="green").save(cwd / "output.png")
        return json.dumps(
            {
                "status": "succeeded",
                "image_files": ["output.png"],
                "error": None,
            }
        )

    payload = GenerationConfirmRequest(
        session_id="sess_test",
        provider="codex_cli_gpt_image",
        mode="t2i",
        original_prompt="城市",
        optimized_prompt="cinematic city",
        parameters={"steps": 28, "guidance": 3.5, "seed": 123},
        reference_image_ids=[],
        codex_reasoning_effort="low",
    )
    provider = CodexImageProvider(settings, executor=fake_executor)

    with new_session(engine) as db:
        job = GenerationJobRecord(
            job_id="job_test",
            session_id=payload.session_id,
            provider=payload.provider,
            mode=payload.mode,
            status="running",
        )
        records = provider.generate(db, job=job, payload=payload, reference_images=[])

    assert len(records) == 1
    assert records[0].image_id.startswith("img_")
    assert records[0].thumbnail_storage_path is not None
    assert records[0].seed == 123
    assert commands[0][commands[0].index("--sandbox") + 1] == "workspace-write"
    assert 'model_reasoning_effort="low"' in commands[0]
    assert commands[0][-1] == "-"
    assert "cinematic city" in prompts[0]
