import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Сборка кладётся прямо в python-пакет (llmlab/static) и коммитится:
// пользователю курса не нужен Node. В dev-режиме API проксируется на `llmlab serve`.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../llmlab/static",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 4000,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
