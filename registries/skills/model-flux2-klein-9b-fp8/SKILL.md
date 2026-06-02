---
name: model-flux2-klein-9b-fp8
description: Use when optimizing prompts or parameters for Local Flux / FLUX.2 Klein 9B FP8, including T2I and I2I.
---

Model notes:
1. Supports text-to-image and image-to-image/reference-based editing.
2. Default distilled setup can start with low inference steps and guidance around 1.0; detailed parameters live in Local Flux settings.
3. Prioritize clear subject, composition, lighting, material, camera, and spatial relations.
4. Avoid overloading the prompt with conflicting styles.
5. For I2I, ask whether to preserve subject identity, composition, color palette, pose, product shape, or only transfer style.
6. Warn that this model may require significant VRAM.
7. Warn that model access may require Hugging Face login and accepting model terms.
8. Warn that text in images may be inaccurate or distorted.
