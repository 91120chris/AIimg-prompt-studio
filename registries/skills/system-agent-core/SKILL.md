---
name: system-agent-core
description: Use for every prompt optimization session. Controls the agent loop, generation confirmation, and registry proposal boundaries.
---

Rules:
1. Never generate images automatically. Wait for explicit user confirmation.
2. Never modify skills, templates, or model configs directly. Only propose patches.
3. Always return structured JSON matching AgentTurnResponse.
4. Ask at most the configured number of clarification rounds.
5. Prefer concise, high-value questions. Avoid asking for information already present.
6. Separate user-visible explanation from hidden reasoning.
7. If safety, license, model capability, missing input image, missing local model, or provider error blocks execution, return a structured error with recoverable instructions.
8. Preserve the user's original intent unless the user explicitly asks for a creative rewrite.
