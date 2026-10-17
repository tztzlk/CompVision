import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Сборка в dist/ (её копирует frontend/Dockerfile в nginx).
// Dev-режим повторяет nginx: /api и /health идут на локальный backend как есть,
// /ui-api переписывается в /api с подстановкой X-API-Key из переменной API_KEY.
// process.env читается только здесь, на сервере разработки: в бандл ключ не попадает
// (в клиентский код Vite пропускает лишь переменные с префиксом VITE_).
const BACKEND = "http://localhost:8000";
const apiKey = process.env.API_KEY ?? "";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/ui-api": {
        target: BACKEND,
        rewrite: (path) => path.replace(/^\/ui-api/, "/api"),
        headers: { "X-API-Key": apiKey },
      },
      "/api": BACKEND,
      "/health": BACKEND,
    },
  },
});
