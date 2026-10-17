import { forwardRef, useMemo, type ReactNode } from "react";
import type { AnalysisResult, FeedbackLabel } from "../api";
import { ViolationCard } from "./ViolationCard";

interface Props {
  result: AnalysisResult;
  onSeek: ((sec: number) => void) | null;
  onOpenFrame: (src: string, caption: string) => void;
  /** Отзыв по нарушению; index — позиция в result.violations (как прислал сервер). */
  onFeedback?: ((index: number, label: FeedbackLabel) => Promise<void>) | null;
  /** Блок между вердиктом и списком (сведения о проверке, превью). */
  children?: ReactNode;
}

/** Вердикт (ровно две формулировки из раздела 4 ТЗ) и список нарушений по таймкоду. */
export const ResultView = forwardRef<HTMLHeadingElement, Props>(function ResultView(
  { result, onSeek, onOpenFrame, onFeedback = null, children },
  headingRef,
) {
  const violations = useMemo(
    () =>
      (result.violations ?? [])
        .map((v, serverIndex) => ({ v, serverIndex }))
        .sort((a, b) => a.v.timecode_sec - b.v.timecode_sec),
    [result.violations],
  );
  // Вердикт берём из поля verdict; при его отсутствии (несогласованный ответ) — по списку.
  const isViolation = result.verdict ? result.verdict === "violation" : violations.length > 0;

  return (
    <section aria-labelledby="verdict-title">
      <div className={isViolation ? "verdict verdict-bad" : "verdict verdict-ok"}>
        <span className="verdict-icon" aria-hidden="true">
          {isViolation ? "!" : "✓"}
        </span>
        <div>
          <h2 id="verdict-title" ref={headingRef} tabIndex={-1} className="verdict-title">
            {isViolation ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}
          </h2>
          {isViolation && (
            <p className="verdict-sub">
              Найдено эпизодов: {violations.length}. Проверьте каждый по кадрам и видео.
            </p>
          )}
        </div>
      </div>

      {children}

      {isViolation && violations.length > 0 && (
        <ol className="cards" aria-label="Возможные нарушения по времени">
          {violations.map(({ v, serverIndex }, i) => (
            <li key={`${v.timecode_sec}-${serverIndex}`}>
              <ViolationCard
                violation={v}
                index={i}
                onSeek={onSeek}
                onOpenFrame={onOpenFrame}
                onFeedback={onFeedback ? (label) => onFeedback(serverIndex, label) : null}
              />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
});
