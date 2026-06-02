import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

const configDir = dirname(fileURLToPath(import.meta.url));
const monorepoRoot = resolve(configDir, "../..");

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, monorepoRoot, "");
  const frontendApiBaseUrl =
    env.FRONTEND_API_BASE_URL ?? env.VITE_FRONTEND_API_BASE_URL ?? "http://127.0.0.1:8000";
  const isBuild = command === "build";

  return {
    plugins: [react()],
    base: isBuild ? "./" : "/",
    clearScreen: false,
    envDir: monorepoRoot,
    define: {
      __FRONTEND_API_BASE_URL__: JSON.stringify(frontendApiBaseUrl),
    },
    resolve: isBuild
      ? {
          alias: {
            "/src": resolve(configDir, "src"),
          },
        }
      : undefined,
    server: {
      host: "127.0.0.1",
      port: 1420,
      strictPort: true,
      watch: {
        ignored: ["**/src-tauri/**"],
      },
    },
  };
});
