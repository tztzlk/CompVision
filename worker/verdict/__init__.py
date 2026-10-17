"""Вердикт: события -> AnalysisResult строго по shared.schemas (раздел 10 ТЗ)."""

from __future__ import annotations

import base64
import logging
import string
from pathlib import Path
from typing import Any, Mapping

from shared.schemas import AnalysisResult, EvidenceFrame, RuleRef, Violation
from worker.events import Event, Rule

__all__ = ["encode_frame", "rule_ref", "render_description", "to_violation", "build_result"]

LOG = logging.getLogger("worker.verdict")

_JPEG_MAGIC = b"\xff\xd8"


def encode_frame(frame_path: Path, timestamp_sec: float) -> EvidenceFrame:
    """Прочитать JPEG и упаковать в EvidenceFrame (base64)."""
    data = Path(frame_path).read_bytes()
    if not data.startswith(_JPEG_MAGIC):
        raise ValueError(f"{frame_path}: не JPEG")
    return EvidenceFrame(
        timestamp_sec=max(0.0, float(timestamp_sec)),
        image_base64=base64.b64encode(data).decode("ascii"),
    )


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


def to_violation(event: Event, rule: Rule, frames_dir: Path) -> Violation:
    """Событие -> Violation: type=rule.title, description из rule.template + ссылка на rule.source."""
    raise NotImplementedError


def build_result(job_id: str, events: list[Event], rules: list[Rule], frames_dir: Path) -> AnalysisResult:
    """Собрать итог: verdict=violation, если есть хотя бы одно событие выше порога, иначе no_violation."""
    raise NotImplementedError
