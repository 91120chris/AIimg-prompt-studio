import { z } from "zod";

export const healthResponseSchema = z.object({
  status: z.literal("ok"),
  app_name: z.string(),
  version: z.string(),
  environment: z.string(),
});

export const codexStatusResponseSchema = z.object({
  provider: z.literal("codex_cli"),
  available: z.boolean(),
  configured_binary: z.string(),
  resolved_kind: z.enum(["native", "cmd", "ps1", "not_found", "unknown"]),
  version: z.string().nullable(),
  warning: z.string().nullable(),
  error: z.string().nullable(),
});

export const codexModelsResponseSchema = z.object({
  default_model: z.string(),
  model_options: z.array(z.string()),
});

export const ollamaStatusResponseSchema = z.object({
  provider: z.literal("ollama_local_llm"),
  available: z.boolean(),
  base_url: z.string(),
  model_count: z.number(),
  error: z.string().nullable(),
});

export const ollamaModelsResponseSchema = z.object({
  selected_model: z.string().nullable(),
  models: z.array(z.string()),
});

export const secretStatusResponseSchema = z.object({
  hf_token_configured: z.boolean(),
  hf_home_configured: z.boolean(),
  hf_hub_cache_configured: z.boolean(),
});

export const structuredErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  suggestion: z.string().nullable(),
});

export const questionOptionSchema = z.object({
  value: z.string(),
  label: z.string(),
  description: z.string().nullable().optional(),
});

export const textQuestionSchema = z.object({
  kind: z.literal("text"),
  question_id: z.string(),
  label: z.string(),
  prompt: z.string(),
  required: z.boolean(),
  placeholder: z.string().nullable().optional(),
  max_length: z.number().nullable().optional(),
});

export const choiceQuestionSchema = z.object({
  kind: z.literal("choice"),
  question_id: z.string(),
  label: z.string(),
  prompt: z.string(),
  required: z.boolean(),
  options: z.array(questionOptionSchema),
  allow_multiple: z.boolean(),
});

export const booleanQuestionSchema = z.object({
  kind: z.literal("boolean"),
  question_id: z.string(),
  label: z.string(),
  prompt: z.string(),
  required: z.boolean(),
  true_label: z.string(),
  false_label: z.string(),
});

export const scaleQuestionSchema = z.object({
  kind: z.literal("scale"),
  question_id: z.string(),
  label: z.string(),
  prompt: z.string(),
  required: z.boolean(),
  min_value: z.number(),
  max_value: z.number(),
  step: z.number(),
});

export const questionSchema = z.discriminatedUnion("kind", [
  textQuestionSchema,
  choiceQuestionSchema,
  booleanQuestionSchema,
  scaleQuestionSchema,
]);

export const questionnaireSchema = z.object({
  questionnaire_id: z.string(),
  title: z.string(),
  description: z.string().nullable().optional(),
  questions: z.array(questionSchema),
});

export const agentTurnResponseSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("message"),
    message: z.string(),
    warnings: z.array(z.string()).optional(),
  }),
  z.object({
    kind: z.literal("questionnaire"),
    message: z.string(),
    questionnaire: questionnaireSchema,
    warnings: z.array(z.string()).optional(),
  }),
  z.object({
    kind: z.literal("optimized_prompt"),
    message: z.string(),
    optimized_prompt: z.string(),
    prompt_version_title: z.string().nullable().optional(),
    warnings: z.array(z.string()).optional(),
  }),
  z.object({
    kind: z.literal("error"),
    error: structuredErrorSchema,
  }),
]);

export const referenceImageResponseSchema = z.object({
  reference_image_id: z.string(),
  session_id: z.string(),
  slot: z.number(),
  role: z.string(),
  url: z.string(),
  thumbnail_url: z.string().nullable(),
  filename: z.string(),
  width: z.number(),
  height: z.number(),
  created_at: z.string(),
});

export const generatedImageResponseSchema = z.object({
  image_id: z.string(),
  session_id: z.string(),
  role: z.string(),
  url: z.string(),
  thumbnail_url: z.string().nullable(),
  filename: z.string(),
  width: z.number(),
  height: z.number(),
  seed: z.number().nullable(),
  provider: z.string(),
  created_at: z.string(),
});

export const generationJobResponseSchema = z.object({
  job_id: z.string(),
  session_id: z.string(),
  provider: z.string(),
  mode: z.string(),
  status: z.enum(["queued", "running", "succeeded", "failed", "cancelled"]),
  images: z.array(generatedImageResponseSchema),
  error: structuredErrorSchema.nullable(),
  created_at: z.string(),
});

export const sessionResponseSchema = z.object({
  session_id: z.string(),
  title: z.string().nullable(),
  created_at: z.string(),
  reference_images: z.array(referenceImageResponseSchema),
  generated_images: z.array(generatedImageResponseSchema),
});

export type HealthResponse = z.infer<typeof healthResponseSchema>;
export type CodexStatusResponse = z.infer<typeof codexStatusResponseSchema>;
export type CodexModelsResponse = z.infer<typeof codexModelsResponseSchema>;
export type OllamaStatusResponse = z.infer<typeof ollamaStatusResponseSchema>;
export type OllamaModelsResponse = z.infer<typeof ollamaModelsResponseSchema>;
export type SecretStatusResponse = z.infer<typeof secretStatusResponseSchema>;
export type AgentTurnResponse = z.infer<typeof agentTurnResponseSchema>;
export type GenerationJobResponse = z.infer<typeof generationJobResponseSchema>;
export type Question = z.infer<typeof questionSchema>;
export type Questionnaire = z.infer<typeof questionnaireSchema>;
export type ReferenceImageResponse = z.infer<typeof referenceImageResponseSchema>;
export type SessionResponse = z.infer<typeof sessionResponseSchema>;
