import type { Violation } from "../api";
import { formatClock, formatSeconds, isHttpUrl } from "../format";

interface Props {
  violation: Violation;
  index: number;
  /** null — превью недоступно, таймкод показывается текстом. */
  onSeek: ((sec: number) => void) | null;
  onOpenFrame: (src: string, caption: string) => void;
}

export function frameSrc(base64: string): string {
  return base64.startsWith("data:") ? base64 : `data:image/jpeg;base64,${base64}`;
}

export function ViolationCard({ violation: v, index, onSeek, onOpenFrame }: Props) {
  const titleId = `violation-${index}-title`;
  const clock = formatClock(v.timecode_sec);
  const frames = Array.isArray(v.evidence_frames) ? v.evidence_frames : [];
  const rule = v.rule ?? null;

  return (
    <article className="card" aria-labelledby={titleId}>
      <header className="card-head">
        <span className="card-num" aria-hidden="true">
          {index + 1}
        </span>
        <h3 id={titleId} className="card-title">
          {v.type}
        </h3>
        <div className="timecode">
          {onSeek ? (
            <button
              type="button"
              className="timecode-btn"
              onClick={() => onSeek(v.timecode_sec)}
              aria-label={`Перейти к ${clock} в превью видео`}
            >
              {clock}
            </button>
          ) : (
            <span className="timecode-text">{clock}</span>
          )}
          <span className="timecode-exact">{formatSeconds(v.timecode_sec)}</span>
        </div>
      </header>

      <dl className="card-body">
        <dt>Что видно и почему это может быть нарушением</dt>
        <dd className="card-desc">{v.description}</dd>

        <dt>Кадры-доказательства</dt>
        <dd>
          {frames.length === 0 ? (
            <span className="muted">Кадры не переданы</span>
          ) : (
            <ul className="frames">
              {frames.map((f, i) => {
                const src = frameSrc(f.image_base64);
                const caption = `${v.type} — кадр ${i + 1}, ${formatClock(f.timestamp_sec)} (${formatSeconds(f.timestamp_sec)})`;
                return (
                  <li key={i} className="frame">
                    <button
                      type="button"
                      className="frame-btn"
                      onClick={() => onOpenFrame(src, caption)}
                      aria-label={`Увеличить кадр ${i + 1}, ${formatClock(f.timestamp_sec)}`}
                    >
                      <img src={src} alt="" />
                    </button>
                    <span className="frame-cap">
                      {formatClock(f.timestamp_sec)} <span className="muted">· {formatSeconds(f.timestamp_sec)}</span>
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </dd>

        <dt>Правило и нормативный источник</dt>
        <dd>
          {rule ? (
            <div className="rule">
              <div>
                <span className="rule-doc">{rule.doc}</span>
                {rule.clause && <span className="rule-clause">, {rule.clause}</span>}
              </div>
              {isHttpUrl(rule.url) && (
                <a className="rule-link" href={rule.url} target="_blank" rel="noopener noreferrer">
                  Открыть источник
                </a>
              )}
            </div>
          ) : (
            <span className="muted">Источник указан в описании</span>
          )}
        </dd>
      </dl>
    </article>
  );
}
