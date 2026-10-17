"""Типы правил и событий (без зависимостей от детекторов, чтобы не было циклического импорта)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["RulesError", "RuleSource", "Rule", "Event"]


class RulesError(ValueError):
    """rules.yaml не прошёл проверку."""


@dataclass(frozen=True)
class RuleSource:
    doc: str
    url: str
    clause: str


@dataclass(frozen=True)
class Rule:
    """Одна запись rules.yaml: id, title, detector, params, source{doc,url,clause}, template."""

    id: str
    title: str
    detector: str
    params: dict[str, Any]
    source: RuleSource
    template: str


@dataclass(frozen=True)
class Event:
    rule_id: str
    timecode_sec: float
    confidence: float
    evidence_timestamps_sec: list[float] = field(default_factory=list)   # 1–3 кадра
    facts: dict[str, Any] = field(default_factory=dict)                 # подстановки в template
