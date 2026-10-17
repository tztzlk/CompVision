import { formatClock, formatSize } from "../format";

interface Props {
  stage: "uploading" | "processing";
  elapsedSec: number;
  fileName: string;
  fileSize: number;
  jobId: string | null;
  serverWarning: boolean;
  onCancel: () => void;
}

export function StatusPanel({ stage, elapsedSec, fileName, fileSize, jobId, serverWarning, onCancel }: Props) {
  return (
    <section className="panel status" aria-labelledby="status-title" aria-busy="true">
      <div className="status-row">
        <span className="spinner" aria-hidden="true" />
        <div className="status-text">
          <h2 id="status-title" className="panel-title">
            {stage === "uploading" ? "Загрузка видео на сервер…" : "Идёт анализ видео…"}
          </h2>
          <p className="muted">
            Прошло <span className="mono">{formatClock(elapsedSec)}</span>. Обработка может занять несколько минут —
            страницу можно не обновлять.
          </p>
        </div>
      </div>
      <dl className="meta">
        <dt>Файл</dt>
        <dd className="file-name-cell">
          {fileName} <span className="muted">({formatSize(fileSize)})</span>
        </dd>
        {jobId && (
          <>
            <dt>Задача</dt>
            <dd className="mono">{jobId}</dd>
          </>
        )}
      </dl>
      {serverWarning && (
        <p className="warning" role="status">
          Сервер недоступен, повторяем запрос…
        </p>
      )}
      <div className="actions">
        <button type="button" className="btn btn-secondary" onClick={onCancel}>
          Новая проверка
        </button>
      </div>
    </section>
  );
}
