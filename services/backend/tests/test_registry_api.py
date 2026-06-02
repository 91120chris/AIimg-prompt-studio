from fastapi.testclient import TestClient

from app.db.models import AppSettingRecord, SkillVersionRecord, TemplateVersionRecord
from app.db.session import new_session
from app.main import create_app
from app.settings import Settings


def make_client(tmp_path) -> tuple[object, TestClient]:
    app = create_app(
        Settings(
            storage_root=str(tmp_path / "storage"),
            database_url=f"sqlite:///{tmp_path / 'app.sqlite3'}",
            _env_file=None,
        )
    )
    return app, TestClient(app)


def test_skill_and_template_lists_return_latest_versions(tmp_path) -> None:
    app, client = make_client(tmp_path)
    with new_session(app.state.engine) as db:
        db.add(
            SkillVersionRecord(
                skill_version_id="skillv_old",
                skill_id="tone",
                content="old",
                created_at="2026-01-01T00:00:00Z",
            )
        )
        db.add(
            SkillVersionRecord(
                skill_version_id="skillv_new",
                skill_id="tone",
                content="new",
                created_at="2026-01-02T00:00:00Z",
            )
        )
        db.add(
            TemplateVersionRecord(
                template_version_id="tmplv_new",
                template_id="product",
                content="template",
            )
        )
        db.commit()

    skills_response = client.get("/skills")
    templates_response = client.get("/templates")
    skill_response = client.get("/skills/tone")
    template_response = client.get("/templates/product")

    assert skills_response.status_code == 200
    tone_skill = next(item for item in skills_response.json() if item["item_id"] == "tone")
    assert tone_skill["content"] == "new"
    assert templates_response.status_code == 200
    assert any(item["item_id"] == "product" for item in templates_response.json())
    assert skill_response.status_code == 200
    assert skill_response.json()["latest_version_id"] == "skillv_new"
    assert template_response.status_code == 200
    assert template_response.json()["content"] == "template"


def test_initial_registries_are_seeded_from_project_files(tmp_path) -> None:
    _, client = make_client(tmp_path)

    skills_response = client.get("/skills")
    templates_response = client.get("/templates")

    assert skills_response.status_code == 200
    assert templates_response.status_code == 200
    assert {item["item_id"] for item in skills_response.json()}.issuperset(
        {
            "system-agent-core",
            "questionnaire-designer",
            "prompt-template-compiler",
            "model-flux2-klein-9b-fp8",
            "model-codex-cli-gpt-image",
            "feedback-refinement",
        }
    )
    assert {item["item_id"] for item in templates_response.json()}.issuperset(
        {
            "advertising-product-hero",
            "academic-educational-image",
            "portrait-photography",
            "ui-typography-mockup",
            "story-illustration",
        }
    )


def test_registry_patch_proposal_approval_creates_new_skill_version(tmp_path) -> None:
    app, client = make_client(tmp_path)
    with new_session(app.state.engine) as db:
        db.add(SkillVersionRecord(skill_version_id="skillv_current", skill_id="tone", content="old"))
        db.commit()

    create_response = client.post(
        "/skills/patch-proposals",
        json={
            "item_id": "tone",
            "diff_text": "--- tone\n+++ tone\n@@\n-old\n+new",
            "proposed_content": "new",
        },
    )
    proposal_id = create_response.json()["proposal_id"]
    approve_response = client.post(f"/skills/patch-proposals/{proposal_id}/approve")
    skill_response = client.get("/skills/tone")

    assert create_response.status_code == 200
    assert create_response.json()["status"] == "pending"
    assert create_response.json()["item_id"] == "tone"
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"
    assert approve_response.json()["applied_version_id"] is not None
    assert skill_response.json()["content"] == "new"


def test_diff_only_registry_patch_proposal_can_be_approved_without_applying(tmp_path) -> None:
    app, client = make_client(tmp_path)
    with new_session(app.state.engine) as db:
        db.add(SkillVersionRecord(skill_version_id="skillv_current", skill_id="tone", content="old"))
        db.commit()

    create_response = client.post(
        "/skills/patch-proposals",
        json={"diff_text": "--- old\n+++ new\n@@\n-old\n+new"},
    )
    proposal_id = create_response.json()["proposal_id"]
    approve_response = client.post(f"/skills/patch-proposals/{proposal_id}/approve")
    skill_response = client.get("/skills/tone")

    assert approve_response.status_code == 200
    assert approve_response.json()["applied_version_id"] is None
    assert skill_response.json()["content"] == "old"


def test_template_patch_proposal_can_be_rejected(tmp_path) -> None:
    _, client = make_client(tmp_path)

    create_response = client.post(
        "/templates/patch-proposals",
        json={"diff_text": "--- template\n+++ template\n@@\n-a\n+b"},
    )
    proposal_id = create_response.json()["proposal_id"]
    reject_response = client.post(f"/templates/patch-proposals/{proposal_id}/reject")

    assert create_response.status_code == 200
    assert create_response.json()["registry_kind"] == "template"
    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "rejected"


def test_template_patch_proposal_approval_creates_new_template_version(tmp_path) -> None:
    app, client = make_client(tmp_path)
    with new_session(app.state.engine) as db:
        db.add(
            TemplateVersionRecord(
                template_version_id="tmplv_current",
                template_id="product",
                content='{"id":"product","name":"Old"}',
            )
        )
        db.commit()

    create_response = client.post(
        "/templates/patch-proposals",
        json={
            "item_id": "product",
            "diff_text": "--- product\n+++ product\n@@\n-Old\n+New",
            "proposed_content": '{"id":"product","name":"New"}',
        },
    )
    proposal_id = create_response.json()["proposal_id"]
    approve_response = client.post(f"/templates/patch-proposals/{proposal_id}/approve")
    template_response = client.get("/templates/product")

    assert create_response.status_code == 200
    assert approve_response.status_code == 200
    assert approve_response.json()["applied_version_id"] is not None
    assert template_response.json()["content"] == '{"id":"product","name":"New"}'


def test_registry_patch_proposal_lists_are_reachable(tmp_path) -> None:
    _, client = make_client(tmp_path)

    skill_create = client.post(
        "/skills/patch-proposals",
        json={"diff_text": "--- skill\n+++ skill\n@@\n-a\n+b"},
    )
    template_create = client.post(
        "/templates/patch-proposals",
        json={"diff_text": "--- template\n+++ template\n@@\n-a\n+b"},
    )

    skills_response = client.get("/skills/patch-proposals")
    templates_response = client.get("/templates/patch-proposals")

    assert skill_create.status_code == 200
    assert template_create.status_code == 200
    assert skills_response.status_code == 200
    assert templates_response.status_code == 200
    assert skills_response.json()[0]["registry_kind"] == "skill"
    assert templates_response.json()[0]["registry_kind"] == "template"


def test_empty_patch_proposal_is_rejected(tmp_path) -> None:
    _, client = make_client(tmp_path)

    response = client.post("/skills/patch-proposals", json={"diff_text": "   "})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "empty_patch_proposal"


def test_patch_proposal_with_content_requires_item_id(tmp_path) -> None:
    _, client = make_client(tmp_path)

    response = client.post(
        "/skills/patch-proposals",
        json={"diff_text": "--- a\n+++ b", "proposed_content": "new content"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "missing_registry_item_id"


def test_skill_enabled_can_be_toggled(tmp_path) -> None:
    _, client = make_client(tmp_path)

    initial = client.get("/skills/questionnaire-designer")
    disabled = client.patch("/skills/questionnaire-designer/enabled", json={"enabled": False})
    after_disable = client.get("/skills/questionnaire-designer")
    enabled = client.patch("/skills/questionnaire-designer/enabled", json={"enabled": True})

    assert initial.status_code == 200
    assert initial.json()["enabled"] is True
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert after_disable.json()["enabled"] is False
    assert enabled.json()["enabled"] is True


def test_template_create_update_duplicate_validate_and_preview(tmp_path) -> None:
    _, client = make_client(tmp_path)
    content = """
    {
      "id": "custom-product",
      "name": "Custom Product",
      "applies_to": ["t2i", "i2i"],
      "description": "A compact product template.",
      "questions": [
        {"id": "subject", "type": "text", "label": "Subject", "required": true},
        {
          "id": "mood",
          "type": "single_choice",
          "label": "Mood",
          "options": ["premium", "playful"],
          "required": false
        },
        {"id": "notes", "type": "textarea", "label": "Notes", "required": false}
      ],
      "prompt_structure": {"must_include": ["subject"], "avoid": []}
    }
    """

    validate_response = client.post("/templates/validate", json={"content": content})
    preview_response = client.post("/templates/preview", json={"content": content})
    create_response = client.post("/templates", json={"content": content})
    updated_content = content.replace("Custom Product", "Updated Product")
    update_response = client.put("/templates/custom-product", json={"content": updated_content})
    duplicate_response = client.post("/templates/custom-product/duplicate", json={})
    list_response = client.get("/templates")

    assert validate_response.status_code == 200
    assert validate_response.json()["valid"] is True
    assert validate_response.json()["template_id"] == "custom-product"
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["valid"] is True
    assert preview["questionnaire"]["questions"][0]["kind"] == "text"
    assert preview["questionnaire"]["questions"][1]["kind"] == "choice"
    assert create_response.status_code == 200
    assert create_response.json()["item_id"] == "custom-product"
    assert update_response.status_code == 200
    assert "Updated Product" in update_response.json()["content"]
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["item_id"].startswith("custom-product-copy")
    assert {"custom-product", duplicate_response.json()["item_id"]}.issubset(
        {item["item_id"] for item in list_response.json()}
    )


def test_template_create_rejects_duplicate_and_invalid_content(tmp_path) -> None:
    _, client = make_client(tmp_path)
    content = """
    {
      "id": "custom-invalid-check",
      "name": "Custom",
      "applies_to": ["t2i"],
      "questions": [{"id": "subject", "type": "text", "label": "Subject"}],
      "prompt_structure": {}
    }
    """

    first = client.post("/templates", json={"content": content})
    duplicate = client.post("/templates", json={"content": content})
    invalid = client.post("/templates/validate", json={"content": "{\"id\":\"broken\"}"})

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "template_id_exists"
    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False


def test_unified_registry_proposal_validation_approval_and_rejection(tmp_path) -> None:
    app, client = make_client(tmp_path)
    template_content = """
    {
      "id": "proposal-template",
      "name": "Proposal Template",
      "applies_to": ["t2i"],
      "description": "Proposal test.",
      "questions": [{"id": "subject", "type": "text", "label": "Subject"}],
      "prompt_structure": {}
    }
    """
    create_response = client.post(
        "/templates/patch-proposals",
        json={
            "item_id": "proposal-template",
            "change_kind": "create",
            "summary": "Add proposal template",
            "diff_text": "Agent proposed create template.",
            "proposed_content": template_content,
        },
    )
    proposal_id = create_response.json()["proposal_id"]
    list_response = client.get("/registry/patch-proposals")
    validate_response = client.post(f"/registry/patch-proposals/{proposal_id}/validate")
    approve_response = client.post(f"/registry/patch-proposals/{proposal_id}/approve")
    template_response = client.get("/templates/proposal-template")

    assert create_response.status_code == 200
    assert create_response.json()["change_kind"] == "create"
    assert list_response.status_code == 200
    assert any(item["proposal_id"] == proposal_id for item in list_response.json())
    assert validate_response.status_code == 200
    assert validate_response.json()["valid"] is True
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"
    assert template_response.status_code == 200

    reject_create = client.post(
        "/skills/patch-proposals",
        json={"item_id": "reject-me", "diff_text": "Reject this proposal."},
    )
    reject_response = client.post(
        f"/registry/patch-proposals/{reject_create.json()['proposal_id']}/reject"
    )

    assert reject_create.status_code == 200
    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "rejected"

    with new_session(app.state.engine) as db:
        record = db.get(TemplateVersionRecord, approve_response.json()["applied_version_id"])
    assert record is not None


def test_unified_registry_proposal_rejects_invalid_template_on_approve(tmp_path) -> None:
    _, client = make_client(tmp_path)
    create_response = client.post(
        "/templates/patch-proposals",
        json={
            "item_id": "invalid-template",
            "change_kind": "create",
            "diff_text": "Invalid template.",
            "proposed_content": '{"id":"invalid-template"}',
        },
    )
    proposal_id = create_response.json()["proposal_id"]
    approve_response = client.post(f"/registry/patch-proposals/{proposal_id}/approve")

    assert create_response.status_code == 200
    assert approve_response.status_code == 422
    assert approve_response.json()["detail"]["code"] == "proposal_invalid"


def test_approved_new_skill_proposal_is_disabled_by_default(tmp_path) -> None:
    app, client = make_client(tmp_path)
    skill_content = "# Product Failure Rule\n\nWhen feedback mentions broken labels, preserve label geometry."
    create_response = client.post(
        "/skills/patch-proposals",
        json={
            "item_id": "product-failure-rule",
            "change_kind": "create",
            "diff_text": "Create new skill.",
            "proposed_content": skill_content,
        },
    )
    approve_response = client.post(
        f"/registry/patch-proposals/{create_response.json()['proposal_id']}/approve"
    )
    skill_response = client.get("/skills/product-failure-rule")

    assert approve_response.status_code == 200
    assert skill_response.status_code == 200
    assert skill_response.json()["content"] == skill_content
    assert skill_response.json()["enabled"] is False
    with new_session(app.state.engine) as db:
        setting = db.get(AppSettingRecord, "skill_enabled:product-failure-rule")
    assert setting is not None
    assert setting.value == "0"
