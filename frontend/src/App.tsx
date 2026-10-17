import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, isAbortError, MOCK_MODE, type AnalysisResult } from "./api";
import { ImageModal } from "./components/ImageModal";
import { ResultView } from "./components/ResultView";
import { StatusPanel } from "./components/StatusPanel";
import { UploadForm } from "./components/UploadForm";
import { VideoPreview, type VideoPreviewHandle } from "./components/VideoPreview";
import { formatClock, formatSize, isMovFile } from "./format";

const POLL_INTERVAL_MS = 2000;
/** Подряд неудачных опросов (сеть/5xx), после которых показываем ошибку. */
const MAX_POLL_FAILURES = 5;

type Retry = "poll" | "upload" | "none";

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; startedAt: number }
  | { kind: "processing"; jobId: string; startedAt: number; attempt: number }
  | { kind: "done"; jobId: string; result: AnalysisResult; elapsedSec: number }
  | { kind: "failed"; jobId: string; elapsedSec: number }
  | {
      kind: "error";
      title: string;
      text: string;
      detail: string | null;
      retry: Retry;
      jobId: string | null;
      startedAt: number | null;
    };

function describeError(e: ApiError, stage: "upload" | "poll"): { title: string; text: string; retry: Retry } {
  switch (e.kind) {
    case "unauthorized":
      return {
        title: "Сервис не настроен",
        text: "Сервер отклонил запрос (401): на сервере не настроен ключ доступа. Обратитесь к администратору сервиса.",
        retry: stage === "poll" ? "poll" : "upload",
      };
    case "too_large":
      return {
        title: "Файл слишком большой",
        text: "Сервер отклонил файл (413): превышен допустимый размер загрузки (по умолчанию 200 МБ). Выберите файл меньшего размера.",
        retry: "none",
      };
    case "unsupported":
      return {
        title: "Файл не того формата",
        text: "Сервер отклонил файл (415): ожидается видео QuickTime (.mov). Выберите другой файл.",
        retry: "none",
      };
    case "not_found":
      return {
        title: "Задача не найдена",
        text: "Сервер не знает эту задачу (404) — возможно, она потеряна. Загрузите видео заново.",
        retry: "none",
      };
    case "unavailable":
      return {
        title: "Сервер недоступен",
        text:
          stage === "poll"
            ? "Не удаётся получить статус задачи: сервер не отвечает. Задача на сервере могла продолжить выполнение — проверьте статус снова."
            : "Не удаётся связаться с сервером анализа. Проверьте подключение и повторите отправку.",
        retry: stage === "poll" ? "poll" : "upload",
      };
    case "bad_response":
      return {
        title: "Некорректный ответ сервера",
        text: "Сервер вернул ответ, который не удалось разобрать.",
        retry: stage === "poll" ? "poll" : "upload",
      };
    default:
      return {
        title: "Запрос отклонён",
        text: `Сервер вернул ошибку${e.status ? ` ${e.status}` : ""}.`,
        retry: "none",
      };
  }
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [serverWarning, setServerWarning] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewOk, setPreviewOk] = useState(true);
  const [modal, setModal] = useState<{ src: string; caption: string } | null>(null);

  const verdictRef = useRef<HTMLHeadingElement>(null);
  const errorRef = useRef<HTMLHeadingElement>(null);
  const previewRef = useRef<VideoPreviewHandle>(null);
  const uploadCtrlRef = useRef<AbortController | null>(null);

  // Локальное превью исходного файла.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPreviewOk(true);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Тикающий секундомер во время загрузки/обработки.
  const active = phase.kind === "uploading" || phase.kind === "processing";
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [active]);

  // Отмена загрузки при размонтировании.
  useEffect(() => () => uploadCtrlRef.current?.abort(), []);

  // Polling статуса задачи. Перезапускается при новом jobId или «Проверить снова» (attempt).
  const pollJobId = phase.kind === "processing" ? phase.jobId : null;
  const pollAttempt = phase.kind === "processing" ? phase.attempt : 0;
  const pollStartedAt = phase.kind === "processing" ? phase.startedAt : 0;
  useEffect(() => {
    if (!pollJobId) return;
    const jobId = pollJobId;
    const ctrl = new AbortController();
    let cancelled = false;
    let timer: number | undefined;
    let failures = 0;

    const schedule = () => {
      if (!cancelled) timer = window.setTimeout(tick, POLL_INTERVAL_MS);
    };
    const elapsed = () => Math.max(0, (Date.now() - pollStartedAt) / 1000);

    async function tick() {
      try {
        const res = await api.getJob(jobId, ctrl.signal);
        if (cancelled) return;
        failures = 0;
        setServerWarning(false);
        if (res.status === "done") {
          setPhase({ kind: "done", jobId, result: res, elapsedSec: elapsed() });
        } else if (res.status === "failed") {
          setPhase({ kind: "failed", jobId, elapsedSec: elapsed() });
        } else {
          schedule();
        }
      } catch (e) {
        if (cancelled || isAbortError(e)) return;
        const err = e instanceof ApiError ? e : new ApiError("unavailable", null, null);
        if (err.kind === "unavailable") {
          failures += 1;
          if (failures < MAX_POLL_FAILURES) {
            setServerWarning(true);
            schedule();
            return;
          }
        }
        setServerWarning(false);
        const d = describeError(err, "poll");
        setPhase({ kind: "error", ...d, detail: err.detail, jobId, startedAt: pollStartedAt });
      }
    }

    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [pollJobId, pollAttempt, pollStartedAt]);

  // Фокус на результат/ошибку при смене экрана.
  useEffect(() => {
    if (phase.kind === "done" || phase.kind === "failed") verdictRef.current?.focus();
    if (phase.kind === "error") errorRef.current?.focus();
  }, [phase.kind]);

  const chooseFile = useCallback((f: File) => {
    setFile(f);
    setPhase({ kind: "idle" });
    if (isMovFile(f)) {
      setFileError(null);
    } else {
      setFileError(
        `Файл «${f.name}» не подходит: нужен видеофайл QuickTime с расширением .mov. Выберите другой файл.`,
      );
    }
  }, []);

  const submit = useCallback(async () => {
    if (!file || fileError) return;
    uploadCtrlRef.current?.abort();
    const ctrl = new AbortController();
    uploadCtrlRef.current = ctrl;
    const startedAt = Date.now();
    setServerWarning(false);
    setPhase({ kind: "uploading", startedAt });
    try {
      const res = await api.analyze(file, ctrl.signal);
      if (ctrl.signal.aborted) return;
      setPhase({ kind: "processing", jobId: res.job_id, startedAt, attempt: 0 });
    } catch (e) {
      if (isAbortError(e) || ctrl.signal.aborted) return;
      const err = e instanceof ApiError ? e : new ApiError("unavailable", null, null);
      const d = describeError(err, "upload");
      setPhase({ kind: "error", ...d, detail: err.detail, jobId: null, startedAt: null });
    } finally {
      if (uploadCtrlRef.current === ctrl) uploadCtrlRef.current = null;
    }
  }, [file, fileError]);

  const reset = useCallback(() => {
    uploadCtrlRef.current?.abort();
    uploadCtrlRef.current = null;
    setServerWarning(false);
    setModal(null);
    setFile(null);
    setFileError(null);
    setPhase({ kind: "idle" });
  }, []);

  const retryPoll = useCallback(() => {
    setPhase((p) =>
      p.kind === "error" && p.jobId
        ? { kind: "processing", jobId: p.jobId, startedAt: p.startedAt ?? Date.now(), attempt: Date.now() }
        : p,
    );
  }, []);

  const openFrame = useCallback((src: string, caption: string) => setModal({ src, caption }), []);
  const closeModal = useCallback(() => setModal(null), []);
  const seek = useCallback((sec: number) => previewRef.current?.seek(sec), []);

  const elapsedSec =
    phase.kind === "uploading" || phase.kind === "processing" ? Math.max(0, (now - phase.startedAt) / 1000) : 0;
  const showPreview = phase.kind === "done" && !!previewUrl && previewOk;

  let liveStatus = "";
  switch (phase.kind) {
    case "idle":
      liveStatus = file && !fileError ? `Выбран файл ${file.name}` : "";
      break;
    case "uploading":
      liveStatus = "Загрузка видео на сервер";
      break;
    case "processing":
      liveStatus = serverWarning ? "Сервер недоступен, повторяем запрос" : "Идёт анализ видео";
      break;
    case "done":
      liveStatus =
        phase.result.verdict === "violation" || (phase.result.violations?.length ?? 0) > 0
          ? "Анализ завершён: обнаружены признаки возможных нарушений"
          : "Анализ завершён: нарушение не обнаружено";
      break;
    case "failed":
      liveStatus = "Обработка не удалась";
      break;
    case "error":
      liveStatus = `Ошибка: ${phase.title}`;
      break;
  }

  return (
    <div className="app">
      {MOCK_MODE && (
        <div className="mock-banner" role="note">
          МОК-режим: сервер не используется, результаты синтетические. Имя файла задаёт сценарий: «clean», «fail»,
          «offline».
        </div>
      )}
      <header className="app-header">
        <div className="container">
          <h1 className="app-title">Проверка кассовой дисциплины по видео</h1>
          <p className="app-sub">Загрузите запись с камеры кассовой зоны — сервис отметит эпизоды для проверки.</p>
        </div>
      </header>

      <main className="container main">
        <p className="disclaimer" role="note">
          <strong>Важно.</strong> Результат — подсказка для проверки человеком, а не окончательное юридическое или
          дисциплинарное заключение. Каждый эпизод требует просмотра видео сотрудником.
        </p>

        <div className="visually-hidden" aria-live="polite" role="status">
          {liveStatus}
        </div>

        {phase.kind === "idle" && (
          <UploadForm file={file} error={fileError} canSubmit onFile={chooseFile} onSubmit={submit} />
        )}

        {(phase.kind === "uploading" || phase.kind === "processing") && file && (
          <StatusPanel
            stage={phase.kind}
            elapsedSec={elapsedSec}
            fileName={file.name}
            fileSize={file.size}
            jobId={phase.kind === "processing" ? phase.jobId : null}
            serverWarning={phase.kind === "processing" && serverWarning}
            onCancel={reset}
          />
        )}

        {phase.kind === "failed" && (
          <section className="panel panel-error" aria-labelledby="failed-title">
            <h2 id="failed-title" ref={verdictRef} tabIndex={-1} className="panel-title">
              Обработка не удалась
            </h2>
            <p>Сервер сообщил, что анализ этого видео завершился неуспешно. Попробуйте загрузить видео ещё раз.</p>
            <dl className="meta">
              <dt>Файл</dt>
              <dd className="file-name-cell">{file?.name ?? "—"}</dd>
              <dt>Задача</dt>
              <dd className="mono">{phase.jobId}</dd>
            </dl>
            <div className="actions">
              <button type="button" className="btn btn-primary" onClick={reset}>
                Новая проверка
              </button>
            </div>
          </section>
        )}

        {phase.kind === "error" && (
          <section className="panel panel-error" aria-labelledby="error-title">
            <h2 id="error-title" ref={errorRef} tabIndex={-1} className="panel-title">
              {phase.title}
            </h2>
            <p>{phase.text}</p>
            {phase.detail && (
              <p className="detail">
                Сообщение сервера: <q>{phase.detail}</q>
              </p>
            )}
            {file && (
              <dl className="meta">
                <dt>Файл</dt>
                <dd className="file-name-cell">
                  {file.name} <span className="muted">({formatSize(file.size)})</span>
                </dd>
                {phase.jobId && (
                  <>
                    <dt>Задача</dt>
                    <dd className="mono">{phase.jobId}</dd>
                  </>
                )}
              </dl>
            )}
            <div className="actions">
              {phase.retry === "poll" && (
                <button type="button" className="btn btn-primary" onClick={retryPoll}>
                  Проверить снова
                </button>
              )}
              {phase.retry === "upload" && file && (
                <button type="button" className="btn btn-primary" onClick={submit}>
                  Отправить снова
                </button>
              )}
              <button
                type="button"
                className={phase.retry === "none" ? "btn btn-primary" : "btn btn-secondary"}
                onClick={reset}
              >
                Новая проверка
              </button>
            </div>
          </section>
        )}

        {phase.kind === "done" && (
          <ResultView
            ref={verdictRef}
            result={phase.result}
            onSeek={showPreview ? seek : null}
            onOpenFrame={openFrame}
            onFeedback={(index, label) => api.sendFeedback(phase.jobId, index, label)}
          >
            <section className="panel result-meta" aria-label="Сведения о проверке">
              <dl className="meta">
                <dt>Файл</dt>
                <dd className="file-name-cell">
                  {file?.name ?? "—"} {file && <span className="muted">({formatSize(file.size)})</span>}
                </dd>
                <dt>Задача</dt>
                <dd className="mono">{phase.jobId}</dd>
                <dt>Время обработки</dt>
                <dd className="mono">{formatClock(phase.elapsedSec)}</dd>
              </dl>
              {showPreview && previewUrl && (
                <VideoPreview
                  ref={previewRef}
                  src={previewUrl}
                  onUnavailable={() => setPreviewOk(false)}
                  withTimecodes={(phase.result.violations?.length ?? 0) > 0}
                />
              )}
              <div className="actions">
                <button type="button" className="btn btn-primary" onClick={reset}>
                  Новая проверка
                </button>
              </div>
            </section>
          </ResultView>
        )}
      </main>

      <footer className="container footer">
        Анализ выполняется локально в контуре банка. Результат требует подтверждения проверяющим.
      </footer>

      {modal && <ImageModal src={modal.src} caption={modal.caption} onClose={closeModal} />}
    </div>
  );
}
