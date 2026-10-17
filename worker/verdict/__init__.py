"""Вердикт: события -> AnalysisResult строго по shared.schemas (раздел 10 ТЗ)."""

from __future__ import annotations

import base64
import logging
import string
import subprocess
from pathlib import Path
from typing import Any, Mapping

from shared.schemas import AnalysisResult, EvidenceFrame, JobStatus, RuleRef, Verdict, Violation
from worker.events import Event, Rule

__all__ = [
    "encode_frame",
    "extract_frame",
    "rule_ref",
    "render_description",
    "to_violation",
    "build_result",
    "select_events",
    "MAX_VIOLATIONS",
    "MAX_FRAMES",
]

LOG = logging.getLogger("worker.verdict")

_JPEG_MAGIC = b"\xff\xd8"
FFMPEG_TIMEOUT_SEC = 60
MAX_VIOLATIONS = 5          # больше нарушений в ответе только мешают проверяющему
MAX_FRAMES = 3              # ограничение схемы: 1–3 кадра на нарушение
DEFAULT_MIN_CONFIDENCE = 0.5


def encode_frame(frame_path: Path, timestamp_sec: float) -> EvidenceFrame:
    """Прочитать JPEG и упаковать в EvidenceFrame (base64)."""
    data = Path(frame_path).read_bytes()
    if not data.startswith(_JPEG_MAGIC):
        raise ValueError(f"{frame_path}: не JPEG")
    return EvidenceFrame(
        timestamp_sec=max(0.0, float(timestamp_sec)),
        image_base64=base64.b64encode(data).decode("ascii"),
    )


def extract_frame(video: Path, timestamp_sec: float, out_path: Path) -> None:
    """Один JPEG-кадр на timestamp_sec через ffmpeg.

    ffmpeg получает путь аргументом процесса, поэтому кириллица в имени файла
    (Х-3 … Х-48) не мешает — в отличие от cv2.imread/imwrite на Windows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, timestamp_sec):.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "3", "-f", "image2", str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC,
            check=False, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg: превышен таймаут {FFMPEG_TIMEOUT_SEC} с") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg завершился с кодом {proc.returncode}: {(proc.stderr or '').strip()[-500:]}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg не создал кадр")


def rule_ref(rule: Rule) -> RuleRef:
    """Правило -> RuleRef для поля Violation.rule."""
    return RuleRef(id=rule.id, doc=rule.source.doc, url=rule.source.url, clause=rule.source.clause)


def _safe_format(template: str, values: Mapping[str, Any]) -> str:
    """str.format без исключений: неизвестный или составной плейсхолдер остаётся как есть,
    неверная спецификация формата -> str(value), битые скобки -> шаблон без подстановок."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        LOG.warning("некорректный template, подстановки пропущены: %r", template)
        return template
    out: list[str] = []
    for literal, field, spec, conv in parsed:
        out.append(literal)
        if field is None:
            continue
        if field not in values:
            raw = "{" + field + (f"!{conv}" if conv else "") + (f":{spec}" if spec else "") + "}"
            LOG.warning("неизвестный плейсхолдер в template: %s", raw)
            out.append(raw)
            continue
        value = values[field]
        if conv == "r":
            value = repr(value)
        elif conv == "s":
            value = str(value)
        try:
            out.append(format(value, spec or ""))
        except (ValueError, TypeError):
            out.append(str(value))
    return "".join(out)


def render_description(rule: Rule, timecode_sec: float, facts: Mapping[str, Any] | None = None) -> str:
    """description по rule.template.

    Плейсхолдеры: {timecode} (секунды, «12.3»), {title}, {doc}, {url}, {clause}, {rule_id}
    и любые ключи из facts (например {what_visible}). Поля правила facts не перекрывают.
    Ссылка на норматив гарантированно попадает в текст: если шаблон не вывел clause/doc,
    они дописываются в конец.
    """
    values: dict[str, Any] = {"timecode": f"{timecode_sec:.1f}"}
    values.update(facts or {})
    values.update(
        title=rule.title,
        doc=rule.source.doc,
        url=rule.source.url,
        clause=rule.source.clause,
        rule_id=rule.id,
    )
    text = _safe_format(rule.template, values).strip()
    if rule.source.clause not in text or rule.source.doc not in text:
        tail = f"Это может нарушать {rule.source.clause} документа «{rule.source.doc}»."
        text = f"{text} {tail}" if text else tail
    return text


def _frame_for(
    timestamp_sec: float, frame_hint: str | None, work_dir: Path, video_path: Path | None
) -> Path | None:
    """Путь к JPEG для таймкода: готовый кадр из work_dir (SceneState.frame), иначе вырезка из видео."""
    if frame_hint:
        candidate = work_dir / frame_hint
        if candidate.is_file():
            return candidate
    if video_path is None:
        return None
    out = work_dir / "frames" / f"evidence_{int(round(timestamp_sec * 1000)):08d}.jpg"
    if out.is_file():
        return out
    try:
        extract_frame(video_path, timestamp_sec, out)
    except RuntimeError as exc:
        LOG.warning("кадр на %.1f с не извлечён: %s", timestamp_sec, exc)
        return None
    return out


def to_violation(event: Event, rule: Rule, work_dir: Path, video_path: Path | None = None) -> Violation | None:
    """Событие -> Violation: type=rule.title, description из rule.template + ссылка на rule.source.

    Кадры: event.evidence_timestamps_sec (не больше MAX_FRAMES, без повторов), для каждого
    подсказка пути из event.facts["frames"] (список той же длины, пути относительно work_dir).
    None — если не удалось получить ни одного кадра (схема требует хотя бы один).
    """
    work_dir = Path(work_dir)
    stamps = list(dict.fromkeys(event.evidence_timestamps_sec or [event.timecode_sec]))[:MAX_FRAMES]
    hints = list(event.facts.get("frames") or [])
    frames: list[EvidenceFrame] = []
    for i, ts in enumerate(stamps):
        path = _frame_for(ts, hints[i] if i < len(hints) else None, work_dir, video_path)
        if path is None:
            continue
        try:
            frames.append(encode_frame(path, ts))
        except (OSError, ValueError) as exc:
            LOG.warning("кадр %s не прочитан: %s", path, exc)
    if not frames:
        LOG.warning("правило %s на %.1f с: нет кадров-доказательств, событие пропущено",
                    rule.id, event.timecode_sec)
        return None
    facts = {k: v for k, v in event.facts.items() if k != "frames"}
    return Violation(
        type=rule.title,
        timecode_sec=round(max(0.0, event.timecode_sec), 1),
        description=render_description(rule, event.timecode_sec, facts),
        evidence_frames=frames,
        rule=rule_ref(rule),
    )


def _min_confidence(rule: Rule) -> float:
    try:
        return float(rule.params.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
    except (TypeError, ValueError):
        LOG.warning("правило %s: некорректный min_confidence, беру %.2f", rule.id, DEFAULT_MIN_CONFIDENCE)
        return DEFAULT_MIN_CONFIDENCE


def select_events(events: list[Event], rules: list[Rule]) -> list[tuple[Event, Rule]]:
    """События выше params.min_confidence своего правила, самые уверенные первыми.

    События неизвестных правил отбрасываются. Используется build_result и scripts/evaluate.py
    (режим --scene-cache, без вырезки кадров).
    """
    by_id = {r.id: r for r in rules}
    kept: list[tuple[Event, Rule]] = []
    for ev in events:
        rule = by_id.get(ev.rule_id)
        if rule is None:
            LOG.warning("событие неизвестного правила %s пропущено", ev.rule_id)
            continue
        if ev.confidence >= _min_confidence(rule):
            kept.append((ev, rule))
    kept.sort(key=lambda er: -er[0].confidence)
    return kept


def build_result(
    job_id: str,
    events: list[Event],
    rules: list[Rule],
    work_dir: Path,
    video_path: Path | None = None,
) -> AnalysisResult:
    """Собрать итог по shared.schemas: verdict=violation ⇔ есть хотя бы одно нарушение.

    События ниже params.min_confidence своего правила и события неизвестных правил отбрасываются.
    Если событий больше MAX_VIOLATIONS, остаются самые уверенные; в ответе — по времени.
    """
    violations: list[Violation] = []
    for ev, rule in select_events(events, rules):
        if len(violations) >= MAX_VIOLATIONS:
            break
        v = to_violation(ev, rule, work_dir, video_path)
        if v is not None:
            violations.append(v)
    violations.sort(key=lambda v: v.timecode_sec)
    return AnalysisResult(
        job_id=job_id,
        status=JobStatus.DONE,
        verdict=Verdict.VIOLATION if violations else Verdict.NO_VIOLATION,
        violations=violations,
    )
