"""Отбор событий в ответ: порог min_confidence правила и лимит числа нарушений.

Отдельно от worker.verdict, чтобы отбор работал без pydantic (scripts/eval_cache.py на машине
без зависимостей сервиса). worker.verdict берёт отсюда же — правило отбора одно.
"""

from __future__ import annotations

import logging

from worker.events.model import Event, Rule

__all__ = ["MAX_VIOLATIONS", "DEFAULT_MIN_CONFIDENCE", "min_confidence", "select_events", "top_events"]

LOG = logging.getLogger("worker.events.selection")

MAX_VIOLATIONS = 5          # больше нарушений в ответе только мешают проверяющему
DEFAULT_MIN_CONFIDENCE = 0.5


def min_confidence(rule: Rule) -> float:
    try:
        return float(rule.params.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
    except (TypeError, ValueError):
        LOG.warning("правило %s: некорректный min_confidence, беру %.2f", rule.id, DEFAULT_MIN_CONFIDENCE)
        return DEFAULT_MIN_CONFIDENCE


def select_events(events: list[Event], rules: list[Rule]) -> list[tuple[Event, Rule]]:
    """События выше params.min_confidence своего правила, самые уверенные первыми.

    События неизвестных правил отбрасываются.
    """
    by_id = {r.id: r for r in rules}
    kept: list[tuple[Event, Rule]] = []
    for ev in events:
        rule = by_id.get(ev.rule_id)
        if rule is None:
            LOG.warning("событие неизвестного правила %s пропущено", ev.rule_id)
            continue
        if ev.confidence >= min_confidence(rule):
            kept.append((ev, rule))
    kept.sort(key=lambda er: -er[0].confidence)
    return kept


def top_events(events: list[Event], rules: list[Rule]) -> list[tuple[Event, Rule]]:
    """То, что попадёт в ответ, если у каждого события найдётся кадр: не больше MAX_VIOLATIONS, по времени."""
    kept = select_events(events, rules)[:MAX_VIOLATIONS]
    return sorted(kept, key=lambda er: er[0].timecode_sec)
