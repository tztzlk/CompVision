import { forwardRef, useId, useState } from "react";

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** Ключ отклонён сервером (401) — нужно ввести заново. */
  rejected: boolean;
}

export const ApiKeyField = forwardRef<HTMLInputElement, Props>(function ApiKeyField(
  { value, onChange, rejected },
  ref,
) {
  const id = useId();
  const hintId = `${id}-hint`;
  const [visible, setVisible] = useState(false);
  return (
    <section className="panel keyfield" aria-labelledby={`${id}-label`}>
      <label id={`${id}-label`} htmlFor={id} className="field-label">
        API-ключ
      </label>
      <div className="keyfield-row">
        <input
          ref={ref}
          id={id}
          className={rejected ? "input input-invalid" : "input"}
          type={visible ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-invalid={rejected || undefined}
          aria-describedby={hintId}
          placeholder="Введите ключ доступа"
        />
        <button type="button" className="btn btn-secondary" onClick={() => setVisible((v) => !v)} aria-pressed={visible}>
          {visible ? "Скрыть" : "Показать"}
        </button>
      </div>
      <p id={hintId} className={rejected ? "hint hint-error" : "hint"}>
        {rejected
          ? "Ключ не принят сервером (401). Введите действующий ключ и повторите."
          : "Хранится только в этой вкладке браузера (sessionStorage) и передаётся в заголовке X-API-Key."}
      </p>
    </section>
  );
});
