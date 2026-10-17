import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Сборка в dist/ (её копирует frontend/Dockerfile в nginx).
// В dev-режиме /api и /health проксируются на локальный backend, как nginx в проде.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
