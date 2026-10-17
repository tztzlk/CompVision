// Типы ответа API строго по shared/schemas.py (раздел 10 ТЗ + необязательное rule).
// Неизвестные поля ответа игнорируются (мы просто их не читаем).

export type JobStatus = "processing" | "done" | "failed";
export type Verdict = "no_violation" | "violation";

export interface EvidenceFrame {
  timestamp_sec: number;
  /** JPEG в base64 БЕЗ префикса data:. */
  image_base64: string;
}

export interface RuleRef {
  id: string;
  doc: string;
  /** Может быть заглушкой вида "TODO: ..." — ссылкой рендерится только http(s). */
  url: string;
  clause: string;
}

export interface Violation {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: EvidenceFrame[];
  rule?: RuleRef | null;
}

export interface AnalysisResult {
  job_id: string;
  status: JobStatus;
  /** Отсутствует, пока status !== "done". */
  verdict?: Verdict | null;
  violations?: Violation[];
}

export interface AnalyzeResponse {
  job_id: string;
  status: JobStatus;
}

/** Класс ошибки, понятный UI. */
export type ApiErrorKind =
  | "unauthorized" // 401: nginx не подставил верный API_KEY (ошибка конфигурации сервера)
  | "too_large" // 413
  | "unsupported" // 415
  | "not_found" // 404
  | "unavailable" // сеть / 5xx
  | "bad_response" // ответ не разобрать
  | "other"; // прочие 4xx

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  /** Текст detail из JSON-тела ошибки, если он был. */
  readonly detail: string | null;

  constructor(kind: ApiErrorKind, status: number | null, detail: string | null, message?: string) {
    super(message ?? detail ?? kind);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

/** Единый интерфейс API-клиента: настоящая реализация (client.ts) и мок (mock.ts). */
export interface ApiClient {
  analyze(file: File, signal?: AbortSignal): Promise<AnalyzeResponse>;
  getJob(jobId: string, signal?: AbortSignal): Promise<AnalysisResult>;
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}
