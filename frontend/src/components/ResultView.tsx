import { forwardRef, useMemo, type ReactNode } from "react";
import type { AnalysisResult } from "../api";
import { ViolationCard } from "./ViolationCard";

interface Props {
  result: AnalysisResult;
  onSeek: ((sec: number) => void) | null;
  onOpenFrame: (src: string, caption: string) => void;
  /** Блок между вердиктом и списком (сведения о проверке, превью). */
  children?: ReactNode;
}

/** Вердикт (ровно две формулировки из раздела 4 ТЗ) и список нарушений по таймкоду. */
export const ResultView = forwardRef<HTMLHeadingElement, Props>(function ResultView(
  { result, onSeek, onOpenFrame, children },
  headingRef,
) {
  const violations = useMemo(
    () => [...(result.violations ?? [])].sort((a, b) => a.timecode_sec - b.timecode_sec),
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
          {violations.map((v, i) => (
            <li key={`${v.timecode_sec}-${i}`}>
              <ViolationCard violation={v} index={i} onSeek={onSeek} onOpenFrame={onOpenFrame} />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
});
