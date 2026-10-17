"""События: применение правил из rules.yaml к временному ряду состояний сцены (SceneTimeline).

detect_events(rules, scene) вызывает для каждого правила детектор по имени rule.detector
из реестра worker.events.detectors.DETECTORS. Неизвестное имя детектора — ошибка уже
в load_rules (worker не стартует); исключение внутри детектора логируется и не роняет задачу.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

from worker.events.model import Event, Rule, RuleSource, RulesError
from worker.scene.contract import SceneTimeline

__all__ = ["RulesError", "RuleSource", "Rule", "Event", "load_rules", "detect_events"]

LOG = logging.getLogger("worker.events")


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

    from worker.events.detectors import DETECTORS

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
        detector = _non_empty_str(f"{where}.detector", raw["detector"])
        if detector not in DETECTORS:
            raise RulesError(f"{where}.detector: неизвестный детектор {detector!r}, есть: {sorted(DETECTORS)}")
        src = _check_keys(f"{where}.source", raw["source"], _SOURCE_KEYS)
        rules.append(
            Rule(
                id=rule_id,
                title=_non_empty_str(f"{where}.title", raw["title"]),
                detector=detector,
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


def detect_events(rules: list[Rule], scene: SceneTimeline) -> list[Event]:
    """Для каждого правила вызвать детектор по имени rule.detector с rule.params.

    Ошибка одного детектора логируется и не мешает остальным. Результат отсортирован по времени.
    """
    from worker.events.detectors import DETECTORS

    events: list[Event] = []
    for rule in rules:
        detector = DETECTORS.get(rule.detector)
        if detector is None:
            LOG.error("правило %s: неизвестный детектор %s", rule.id, rule.detector)
            continue
        try:
            found = detector(scene, rule.params)
        except Exception:  # noqa: BLE001 — один сломанный детектор не роняет задачу
            LOG.exception("правило %s: ошибка детектора %s", rule.id, rule.detector)
            continue
        events.extend(dataclasses.replace(ev, rule_id=rule.id) for ev in found)
    events.sort(key=lambda ev: ev.timecode_sec)
    return events
