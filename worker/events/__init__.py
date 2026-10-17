"""События: применение правил из rules.yaml к последовательности детекций."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from worker.perception import FrameDetections

__all__ = ["RulesError", "RuleSource", "Rule", "Event", "load_rules", "detect_events"]


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


_RULE_KEYS = {"id", "title", "detector", "params", "source", "template"}
_SOURCE_KEYS = {"doc", "url", "clause"}


def _check_keys(where: str, data: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RulesError(f"{where}: ожидается словарь, получено {type(data).__name__}")
    keys = set(data)
    missing, extra = expected - keys, keys - expected
    if missing:
        raise RulesError(f"{where}: нет обязательных полей {sorted(missing)}")
    if extra:
        raise RulesError(f"{where}: неизвестные поля {sorted(map(str, extra))}")
    return data


def _non_empty_str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RulesError(f"{where}: ожидается непустая строка")
    return value.strip()


def load_rules(rules_path: Path) -> list[Rule]:
    """Прочитать и строго провалидировать rules.yaml (это вход: лишние/недостающие поля — ошибка).

    Формат: {"rules": [ {id, title, detector, params, source{doc,url,clause}, template}, ... ]}.
    """
    try:
        with Path(rules_path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise RulesError(f"не удалось прочитать {rules_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RulesError(f"{rules_path}: некорректный YAML: {exc}") from exc

    top = _check_keys(str(rules_path), data, {"rules"})
    items = top["rules"]
    if not isinstance(items, list) or not items:
        raise RulesError(f"{rules_path}: rules должен быть непустым списком")

    rules: list[Rule] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        where = f"{rules_path}: rules[{i}]"
        raw = _check_keys(where, item, _RULE_KEYS)
        rule_id = _non_empty_str(f"{where}.id", raw["id"])
        if rule_id in seen:
            raise RulesError(f"{where}: повторяющийся id {rule_id!r}")
        seen.add(rule_id)
        params = raw["params"]
        if params is None:
            params = {}
        if not isinstance(params, dict) or not all(isinstance(k, str) for k in params):
            raise RulesError(f"{where}.params: ожидается словарь со строковыми ключами")
        src = _check_keys(f"{where}.source", raw["source"], _SOURCE_KEYS)
        rules.append(
            Rule(
                id=rule_id,
                title=_non_empty_str(f"{where}.title", raw["title"]),
                detector=_non_empty_str(f"{where}.detector", raw["detector"]),
                params=dict(params),
                source=RuleSource(
                    doc=_non_empty_str(f"{where}.source.doc", src["doc"]),
                    url=_non_empty_str(f"{where}.source.url", src["url"]),
                    clause=_non_empty_str(f"{where}.source.clause", src["clause"]),
                ),
                template=_non_empty_str(f"{where}.template", raw["template"]),
            )
        )
    return rules


def detect_events(rules: list[Rule], detections: list[FrameDetections]) -> list[Event]:
    """Для каждого правила вызвать детектор по имени rule.detector с rule.params."""
    raise NotImplementedError
