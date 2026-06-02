---
name: prompt-template-compiler
description: Compile the user prompt, questionnaire answers, selected template, and model adapter rules into optimized prompts and generation parameters.
---

Output:
- optimized prompt
- concise rationale
- generation parameters when known
- warnings
- optional registry patch proposals

Rules:
1. Preserve user intent.
2. Do not invent critical facts such as brand, person identity, medical/scientific claims, exact text, logo, or product identity unless the user provided them.
3. Prefer concrete visual descriptions over vague adjectives.
4. Make prompt wording model-aware.
5. Avoid contradictory style instructions.
6. For I2I, clearly separate preserve instructions from change instructions.
7. If the selected model has weak text rendering, warn the user before generating text-heavy images.
