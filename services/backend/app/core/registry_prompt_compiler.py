import json


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_registry_proposal_prompt(
    *,
    proposal_kind: str,
    change_kind: str,
    authoring_instruction: str,
    context: dict[str, object],
) -> str:
    target_description = (
        "a registry template JSON object"
        if proposal_kind == "template"
        else "a Markdown SKILL.md document"
    )
    return f"""You are Prompt Optimizer Studio's registry authoring agent.
Create {target_description} as a reviewable proposal. Do not describe changes outside the output schema.

Return only an AgentTurnResponse JSON object with:
- kind: "optimized_prompt"
- message: a short Traditional Chinese summary
- prompt_version_title: a concise proposal title
- optimized_prompt: the exact proposed {target_description} content
- warnings: []

Proposal request:
- proposal_kind: {proposal_kind}
- change_kind: {change_kind}
- authoring_instruction: {authoring_instruction}

Rules:
- Never include secrets or local private filesystem paths.
- For template proposals, optimized_prompt must be a JSON object string with id, name, applies_to, description, questions, and prompt_structure.
- For skill proposals, optimized_prompt must be Markdown suitable for SKILL.md and start with a heading.
- Use the selected template and enabled skills as context, but do not silently mutate them.
- The user must approve the proposal in Manager before it becomes official.

Safe current context:
{_json_dump(context)}
"""
