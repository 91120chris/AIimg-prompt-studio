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
  default_reasoning_effort: z.enum(["low", "medium", "high", "xhigh"]).default("medium"),
  reasoning_effort_options: z
    .array(z.enum(["low", "medium", "high", "xhigh"]))
    .default(["low", "medium", "high", "xhigh"]),
  default_reasoning_summary: z.enum(["auto", "concise", "detailed", "none"]).default("auto"),
  default_verbosity: z.enum(["low", "medium", "high"]).nullable().default(null),
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

export const safeSettingsResponseSchema = z.object({
  app_name: z.string(),
  app_version: z.string(),
  app_env: z.string(),
  backend_host: z.string(),
  backend_port: z.number(),
  selected_agent_provider: z.enum(["codex_cli", "ollama_local_llm"]),
  selected_image_provider: z.enum(["codex_cli_gpt_image", "local_flux"]),
  cors_allow_origins: z.array(z.string()),
  codex_binary_path: z.string(),
  codex_default_model: z.string(),
  codex_model_options: z.array(z.string()),
  codex_default_reasoning_effort: z.string(),
  codex_reasoning_effort_options: z.array(z.string()),
  codex_default_reasoning_summary: z.string(),
  codex_default_verbosity: z.string().nullable(),
  codex_timeout_seconds: z.number(),
  run_codex_smoke: z.boolean(),
  ollama_base_url: z.string(),
  ollama_selected_model: z.string().nullable(),
  ollama_timeout_seconds: z.number(),
  ollama_agent_temperature: z.number(),
  hf_home_configured: z.boolean(),
  hf_hub_cache_configured: z.boolean(),
  frontend_api_base_url: z.string(),
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

export const sessionsResponseSchema = z.array(sessionResponseSchema);

export const registryItemResponseSchema = z.object({
  registry_kind: z.enum(["skill", "template"]),
  item_id: z.string(),
  latest_version_id: z.string().nullable(),
  content: z.string(),
  enabled: z.boolean().nullable().optional(),
  created_at: z.string().nullable(),
});

export const registryPatchProposalResponseSchema = z.object({
  proposal_id: z.string(),
  registry_kind: z.enum(["skill", "template"]),
  item_id: z.string().nullable(),
  status: z.enum(["pending", "approved", "rejected"]),
  diff_text: z.string(),
  proposed_content: z.string().nullable(),
  applied_version_id: z.string().nullable(),
  created_at: z.string(),
});

export const modelInfoResponseSchema = z.object({
  provider: z.string(),
  label: z.string(),
  status: z.string(),
  installed: z.boolean(),
  path_configured: z.boolean(),
  path_label: z.string().nullable(),
  message: z.string().nullable(),
});

export const localFluxStatusResponseSchema = z.object({
  provider: z.literal("local_flux"),
  available: z.boolean(),
  base_url: z.string(),
  message: z.string(),
  error: z.string().nullable(),
});

export const localFluxSettingsResponseSchema = z.object({
  provider: z.literal("local_flux"),
  base_url: z.string(),
  workflow_path: z.string(),
  i2i_one_workflow_path: z.string(),
  i2i_two_workflow_path: z.string(),
  model_path: z.string(),
  vae_path: z.string(),
  text_encoder_path: z.string(),
  width: z.number(),
  height: z.number(),
  seed: z.number().nullable(),
  steps: z.number(),
  cfg: z.number(),
  sampler_name: z.string(),
  scheduler: z.string(),
  denoise: z.number(),
  guidance: z.number(),
  output_prefix: z.string(),
  timeout_seconds: z.number(),
});

export const localFluxWorkflowValidationResponseSchema = z.object({
  valid: z.boolean(),
  workflow_path: z.string(),
  workflow_format: z.enum(["api", "ui", "unknown"]),
  missing_bindings: z.array(z.string()),
  message: z.string(),
});

export const templateValidationResponseSchema = z.object({
  valid: z.boolean(),
  template_id: z.string().nullable(),
  name: z.string().nullable(),
  errors: z.array(z.string()),
});

export const templatePreviewResponseSchema = z.object({
  valid: z.boolean(),
  template_id: z.string().nullable(),
  questionnaire: questionnaireSchema.nullable(),
  errors: z.array(z.string()),
});

export const logResponseSchema = z.object({
  log_id: z.string(),
  level: z.string(),
  message: z.string(),
  created_at: z.string(),
});

export const registryItemsResponseSchema = z.array(registryItemResponseSchema);
export const registryPatchProposalsResponseSchema = z.array(registryPatchProposalResponseSchema);
export const modelInfoListResponseSchema = z.array(modelInfoResponseSchema);
export const logsResponseSchema = z.array(logResponseSchema);

export type HealthResponse = z.infer<typeof healthResponseSchema>;
export type CodexStatusResponse = z.infer<typeof codexStatusResponseSchema>;
export type CodexModelsResponse = z.infer<typeof codexModelsResponseSchema>;
export type OllamaStatusResponse = z.infer<typeof ollamaStatusResponseSchema>;
export type OllamaModelsResponse = z.infer<typeof ollamaModelsResponseSchema>;
export type SecretStatusResponse = z.infer<typeof secretStatusResponseSchema>;
export type SafeSettingsResponse = z.infer<typeof safeSettingsResponseSchema>;
export type AgentTurnResponse = z.infer<typeof agentTurnResponseSchema>;
export type GenerationJobResponse = z.infer<typeof generationJobResponseSchema>;
export type Question = z.infer<typeof questionSchema>;
export type Questionnaire = z.infer<typeof questionnaireSchema>;
export type ReferenceImageResponse = z.infer<typeof referenceImageResponseSchema>;
export type SessionResponse = z.infer<typeof sessionResponseSchema>;
export type RegistryItemResponse = z.infer<typeof registryItemResponseSchema>;
export type RegistryPatchProposalResponse = z.infer<typeof registryPatchProposalResponseSchema>;
export type TemplateValidationResponse = z.infer<typeof templateValidationResponseSchema>;
export type TemplatePreviewResponse = z.infer<typeof templatePreviewResponseSchema>;
export type ModelInfoResponse = z.infer<typeof modelInfoResponseSchema>;
export type LocalFluxStatusResponse = z.infer<typeof localFluxStatusResponseSchema>;
export type LocalFluxSettingsResponse = z.infer<typeof localFluxSettingsResponseSchema>;
export type LocalFluxWorkflowValidationResponse = z.infer<
  typeof localFluxWorkflowValidationResponseSchema
>;
export type LogResponse = z.infer<typeof logResponseSchema>;
