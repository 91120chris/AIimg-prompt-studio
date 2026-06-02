---
name: questionnaire-designer
description: Generate dynamic questionnaires for T2I/I2I prompt optimization using selected template, model constraints, and missing information.
---

Rules:
1. Identify missing visual information: subject, composition, setting, style, lighting, camera, mood, constraints, text/logo, aspect ratio, and reference image role.
2. Ask only questions that materially improve generation.
3. Use supported question types: text, choice, boolean, and scale.
4. Include a final optional free-note field when useful.
5. Keep one questionnaire under 6 questions unless expert mode is enabled.
6. For I2I, ask what should be preserved and what should change.
7. For text/logo-heavy images, ask for exact text and warn that image models may render text imperfectly.
