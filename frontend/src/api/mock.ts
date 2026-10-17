// Моковый API-клиент для демонстрации без сервера. Тот же интерфейс ApiClient,
// что и у настоящего клиента: компоненты не знают, с какой реализацией работают.
//
// Сценарий выбирается по имени файла:
//   содержит "clean"   -> done / no_violation
//   содержит "fail"    -> failed
//   содержит "offline" -> сервер «недоступен» во время опроса статуса
//   иначе              -> done / violation, 3 нарушения
// Ключ "invalid" имитирует 401. Задержка обработки ~6 с (через processing).
import {
  ApiError,
  type AnalysisResult,
  type ApiClient,
  type EvidenceFrame,
  type Violation,
} from "./types";

type Scenario = "clean" | "fail" | "offline" | "violations";

interface MockJob {
  id: string;
  scenario: Scenario;
  createdAt: number;
  result: AnalysisResult | null;
}

const PROCESSING_MS = 6000;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const t = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      window.clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function scenarioFor(name: string): Scenario {
  const n = name.toLowerCase();
  if (n.includes("clean")) return "clean";
  if (n.includes("fail")) return "fail";
  if (n.includes("offline")) return "offline";
  return "violations";
}

function randomId(): string {
  let s = "";
  for (let i = 0; i < 12; i++) s += Math.floor(Math.random() * 16).toString(16);
  return s;
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Синтетический кадр: JPEG base64 без префикса data:, как отдаёт API. */
function makeFrame(sec: number, label: string, hue: number): EvidenceFrame {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const g = ctx.createLinearGradient(0, 0, 640, 360);
    g.addColorStop(0, `hsl(${hue}, 12%, 28%)`);
    g.addColorStop(1, `hsl(${hue}, 12%, 14%)`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 640, 360);
    // «стойка кассира»
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(0, 250, 640, 110);
    // рамка «обнаруженного объекта»
    ctx.strokeStyle = "#e5484d";
    ctx.lineWidth = 4;
    const x = 140 + ((sec * 37) % 260);
    ctx.strokeRect(x, 110, 170, 150);
    ctx.fillStyle = "#e5484d";
    ctx.font = "bold 18px sans-serif";
    ctx.fillText(label, x, 100);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 22px monospace";
    ctx.fillText(`CAM-01  ${fmt(sec)}  (${sec.toFixed(1)} с)`, 16, 34);
    ctx.font = "16px sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("МОК-кадр: синтетическое изображение", 16, 340);
  }
  const url = canvas.toDataURL("image/jpeg", 0.8);
  return { timestamp_sec: sec, image_base64: url.slice(url.indexOf(",") + 1) };
}

function buildViolations(): Violation[] {
  // Порядок намеренно не по таймкоду — UI обязан отсортировать.
  return [
    {
      type: "Пересчёт наличных вне зоны обзора камеры",
      timecode_sec: 47.6,
      description:
        "Кассир переносит пачку банкнот под стойку и пересчитывает её вне поля зрения камеры (≈6 с). " +
        "Это может нарушать п. 12 Правил ведения кассовых операций (тестовые данные).",
      evidence_frames: [makeFrame(46.0, "банкноты", 210), makeFrame(47.6, "руки вне обзора", 210), makeFrame(49.2, "возврат", 210)],
      rule: {
        id: "cash_out_of_view",
        doc: "Правила ведения кассовых операций (тестовые данные)",
        url: "TODO: уточнить ссылку на adilet.zan.kz",
        clause: "п. 12",
      },
    },
    {
      type: "Посторонний в кассовой зоне",
      timecode_sec: 12.4,
      description:
        "В помещение кассы входит человек без форменной одежды и находится у стойки около 20 с. " +
        "Это может нарушать п. 5 Требований к организации кассового узла (тестовые данные).",
      evidence_frames: [makeFrame(12.4, "посторонний", 30), makeFrame(18.0, "у стойки", 30)],
      rule: {
        id: "stranger_in_cash_zone",
        doc: "Требования к организации кассового узла (тестовые данные)",
        url: "https://adilet.zan.kz/rus",
        clause: "п. 5",
      },
    },
    {
      type: "Касса оставлена без присмотра",
      timecode_sec: 95.0,
      description:
        "Кассир покидает рабочее место, денежный ящик остаётся открытым около 40 с. " +
        "Это может нарушать п. 8 регламента кассового обслуживания (тестовые данные).",
      evidence_frames: [makeFrame(95.0, "открытый ящик", 120)],
    },
  ];
}

export function createMockClient(): ApiClient {
  const jobs = new Map<string, MockJob>();

  function checkKey(apiKey: string) {
    if (!apiKey || apiKey.trim().toLowerCase() === "invalid") {
      throw new ApiError("unauthorized", 401, "Неверный или отсутствующий API-ключ (мок)");
    }
  }

  return {
    async analyze(file, apiKey, signal) {
      await sleep(600, signal);
      checkKey(apiKey);
      const id = randomId();
      jobs.set(id, { id, scenario: scenarioFor(file.name), createdAt: Date.now(), result: null });
      return { job_id: id, status: "processing" };
    },

    async getJob(jobId, apiKey, signal) {
      await sleep(250, signal);
      const job = jobs.get(jobId);
      if (job?.scenario === "offline") {
        throw new ApiError("unavailable", 503, null, "mock offline");
      }
      checkKey(apiKey);
      if (!job) throw new ApiError("not_found", 404, "Задача не найдена (мок)");
      if (Date.now() - job.createdAt < PROCESSING_MS) {
        return { job_id: jobId, status: "processing", violations: [] };
      }
      if (!job.result) {
        if (job.scenario === "fail") {
          job.result = { job_id: jobId, status: "failed", violations: [] };
        } else if (job.scenario === "clean") {
          job.result = { job_id: jobId, status: "done", verdict: "no_violation", violations: [] };
        } else {
          job.result = { job_id: jobId, status: "done", verdict: "violation", violations: buildViolations() };
        }
      }
      return job.result;
    },
  };
}
