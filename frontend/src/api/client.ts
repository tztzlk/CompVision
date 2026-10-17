// Настоящий HTTP-клиент по разделу 9 контракта.
import {
  ApiError,
  type AnalysisResult,
  type AnalyzeResponse,
  type ApiClient,
  type ApiErrorKind,
  isAbortError,
} from "./types";

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "/api";

function joinUrl(base: string, path: string): string {
  return base.replace(/\/+$/, "") + path;
}

/** Достаёт detail из JSON-тела ошибки; HTML (например, 413 от nginx) игнорирует. */
function extractDetail(body: string): string | null {
  const text = body.trim();
  if (!text.startsWith("{")) return null;
  try {
    const data: unknown = JSON.parse(text);
    if (data && typeof data === "object" && "detail" in data) {
      const detail = (data as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        // FastAPI 422: [{msg: "..."}]
        const msgs = detail
          .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : ""))
          .filter(Boolean);
        return msgs.length ? msgs.join("; ") : null;
      }
    }
  } catch {
    /* не JSON */
  }
  return null;
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 401) return "unauthorized";
  if (status === 404) return "not_found";
  if (status === 409) return "conflict";
  if (status === 413) return "too_large";
  if (status === 415) return "unsupported";
  if (status >= 500) return "unavailable";
  return "other";
}

async function request<T>(url: string, init: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(url, init);
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError("unavailable", null, null, "network error");
  }
  if (!resp.ok) {
    let body = "";
    try {
      body = await resp.text();
    } catch (e) {
      if (isAbortError(e)) throw e;
    }
    throw new ApiError(kindForStatus(resp.status), resp.status, extractDetail(body));
  }
  try {
    return (await resp.json()) as T;
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError("bad_response", resp.status, null, "invalid JSON");
  }
}

export function createHttpClient(base: string = API_BASE_URL): ApiClient {
  return {
    async analyze(file, apiKey, signal) {
      const form = new FormData();
      form.append("file", file, file.name);
      const data = await request<AnalyzeResponse>(joinUrl(base, "/v1/analyze"), {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: form,
        signal,
      });
      if (!data || typeof data.job_id !== "string" || !data.job_id) {
        throw new ApiError("bad_response", 200, null, "no job_id");
      }
      return data;
    },

    async getJob(jobId, apiKey, signal) {
      const data = await request<AnalysisResult>(
        joinUrl(base, `/v1/jobs/${encodeURIComponent(jobId)}`),
        { method: "GET", headers: { "X-API-Key": apiKey, Accept: "application/json" }, signal, cache: "no-store" },
      );
      if (!data || typeof data.status !== "string") {
        throw new ApiError("bad_response", 200, null, "no status");
      }
      return data;
    },

    async sendFeedback(jobId, violationIndex, label, apiKey) {
      await request<{ status: string }>(joinUrl(base, `/v1/jobs/${encodeURIComponent(jobId)}/feedback`), {
        method: "POST",
        headers: { "X-API-Key": apiKey, "Content-Type": "application/json" },
        body: JSON.stringify({ violation_index: violationIndex, label }),
      });
    },
  };
}
