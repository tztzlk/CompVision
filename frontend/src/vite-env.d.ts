/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Базовый URL API; задаётся build-arg VITE_API_BASE_URL (по умолчанию /api). */
  readonly VITE_API_BASE_URL?: string;
  /** "1" — моковый режим без сервера (демо). Ключи API сюда не запекать. */
  readonly VITE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
