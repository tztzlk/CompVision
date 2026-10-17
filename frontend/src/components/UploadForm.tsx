import { useId, useRef, useState, type DragEvent } from "react";
import { formatSize } from "../format";

interface Props {
  file: File | null;
  error: string | null;
  canSubmit: boolean;
  onFile: (file: File) => void;
  onSubmit: () => void;
}

export function UploadForm({ file, error, canSubmit, onFile, onSubmit }: Props) {
  const inputId = useId();
  const errorId = `${inputId}-error`;
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  function pick(files: FileList | null) {
    const f = files?.[0];
    if (f) onFile(f);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    pick(e.dataTransfer.files);
  }

  return (
    <section className="panel" aria-labelledby={`${inputId}-title`}>
      <h2 id={`${inputId}-title`} className="panel-title">
        Видео для проверки
      </h2>
      <div
        className={`dropzone${dragOver ? " dropzone-over" : ""}${error ? " dropzone-error" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
          setDragOver(true);
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={(e) => {
          if (e.currentTarget === e.target) setDragOver(false);
        }}
        onDrop={onDrop}
      >
        <p className="dropzone-text">Перетащите файл .mov сюда</p>
        <p className="dropzone-or">или</p>
        <label htmlFor={inputId} className="btn btn-secondary">
          Выбрать файл…
        </label>
        <input
          ref={inputRef}
          id={inputId}
          className="visually-hidden"
          type="file"
          accept=".mov,.MOV,video/quicktime"
          aria-describedby={error ? errorId : undefined}
          onChange={(e) => {
            pick(e.target.files);
            // Позволяет выбрать тот же файл повторно.
            e.target.value = "";
          }}
        />
        <p className="hint">Формат: видео QuickTime (.mov)</p>
      </div>

      {error && (
        <p id={errorId} className="inline-error" role="alert">
          {error}
        </p>
      )}

      {file && !error && (
        <div className="file-row">
          <div className="file-info">
            <span className="file-name" title={file.name}>
              {file.name}
            </span>
            <span className="file-size">{formatSize(file.size)}</span>
          </div>
          <button type="button" className="btn btn-primary" onClick={onSubmit} disabled={!canSubmit}>
            Проверить видео
          </button>
        </div>
      )}
      {file && !error && !canSubmit && <p className="hint">Чтобы начать проверку, введите API-ключ.</p>}
    </section>
  );
}
