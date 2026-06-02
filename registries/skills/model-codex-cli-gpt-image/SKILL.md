---
name: model-codex-cli-gpt-image
description: Use when optimizing prompts or generation/editing plans for Codex CLI image generation.
---

Rules:
1. Codex CLI image generation is the default image provider.
2. Do not require an OpenAI API key.
3. For T2I, generate from the optimized prompt only after user confirmation.
4. For I2I, attach reference images separately; do not insert file paths into the prompt.
5. For editing, specify what must remain unchanged and what should change.
6. Ask for exact text, logo, brand, or layout requirements when relevant.
7. Save generated files into the current session generated folder.
8. Return structured GenerationResult JSON.
