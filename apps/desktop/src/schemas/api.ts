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
export type ReferenceImageResponse = z.infer<typeof referenceImageResponseSchema>;
export type SessionResponse = z.infer<typeof sessionResponseSchema>;
