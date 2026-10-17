// Выбор реализации API-клиента: единственное место, где известно о моке.
import { createHttpClient } from "./client";
import { createMockClient } from "./mock";
import type { ApiClient } from "./types";

function detectMock(): boolean {
  if (import.meta.env.VITE_MOCK === "1") return true;
  try {
    return new URLSearchParams(window.location.search).get("mock") === "1";
  } catch {
    return false;
  }
}

export const MOCK_MODE: boolean = detectMock();
export const api: ApiClient = MOCK_MODE ? createMockClient() : createHttpClient();
export * from "./types";
