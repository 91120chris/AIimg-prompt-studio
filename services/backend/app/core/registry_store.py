import json
from pathlib import Path

from sqlmodel import Session, select

from app.core.session_workspace import new_id
from app.db.models import (
    RegistryPatchProposalRecord,
    SkillVersionRecord,
    TemplateVersionRecord,
)
from app.db.session import new_session


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def registry_root() -> Path:
    return project_root() / "registries"


def seed_initial_registries(engine, root: Path | None = None) -> None:
    base = root or registry_root()
    if not base.exists():
        return

    with new_session(engine) as db:
        _seed_skills(db, base / "skills")
        _seed_templates(db, base / "templates")
        db.commit()


def apply_registry_patch_proposal(
    db: Session,
    record: RegistryPatchProposalRecord,
) -> str | None:
    if record.applied_version_id:
        return record.applied_version_id
    if not record.target_id or record.proposed_content is None:
        return None

    if record.registry_kind == "skill":
        version_id = new_id("skillv")
        db.add(
            SkillVersionRecord(
                skill_version_id=version_id,
                skill_id=record.target_id,
                content=record.proposed_content,
            )
        )
    elif record.registry_kind == "template":
        version_id = new_id("tmplv")
        db.add(
            TemplateVersionRecord(
                template_version_id=version_id,
                template_id=record.target_id,
                content=record.proposed_content,
            )
        )
    else:
        return None

    record.applied_version_id = version_id
    return version_id


def _seed_skills(db: Session, skills_root: Path) -> None:
    if not skills_root.exists():
        return

    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        skill_id = skill_file.parent.name
        if _skill_exists(db, skill_id):
            continue
        db.add(
            SkillVersionRecord(
                skill_version_id=f"skillv_seed_{skill_id}",
                skill_id=skill_id,
                content=skill_file.read_text(encoding="utf-8").strip(),
            )
        )


def _seed_templates(db: Session, templates_root: Path) -> None:
    if not templates_root.exists():
        return

    for template_file in sorted(templates_root.glob("*.json")):
        content = template_file.read_text(encoding="utf-8").strip()
        template_id = _template_id(template_file, content)
        if _template_exists(db, template_id):
            continue
        db.add(
            TemplateVersionRecord(
                template_version_id=f"tmplv_seed_{template_id}",
                template_id=template_id,
                content=content,
            )
        )


def _template_id(path: Path, content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return path.stem
    value = payload.get("id") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value.strip() else path.stem


def _skill_exists(db: Session, skill_id: str) -> bool:
    return (
        db.exec(select(SkillVersionRecord).where(SkillVersionRecord.skill_id == skill_id)).first()
        is not None
    )


def _template_exists(db: Session, template_id: str) -> bool:
    return (
        db.exec(
            select(TemplateVersionRecord).where(TemplateVersionRecord.template_id == template_id)
        ).first()
        is not None
    )
