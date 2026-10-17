"""Детекторы правил: SceneTimeline + params -> список Event.

Реестр DETECTORS = {имя из rules.yaml: функция(scene, params) -> list[Event]}.
Детектор возвращает события с пустым rule_id — его проставляет events.detect_events.

Общая механика (одинаковая для всех правил на вероятностях):
  1. из состояний строится ряд «вероятность нарушения» p(t) (комбинация полей контракта);
     None — нет данных: ни нарушение, ни норма;
  2. скользящая медиана по smooth_points точкам (None пропускаются);
  3. гистерезис: интервал начинается при p >= p_on и заканчивается при p < p_off;
     None не меняет состояние и не продлевает интервал;
  4. интервалы короче min_duration_sec отбрасываются, ближе merge_gap_sec — склеиваются;
  5. таймкод = начало интервала, confidence = средняя p на интервале,
     кадры: начало, +1 с, момент максимальной p (не больше трёх, без повторов).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Callable, Mapping, Optional, Sequence

from worker.events.model import Event
from worker.scene.contract import BBox, SceneState, SceneTimeline

__all__ = ["DETECTORS", "Interval", "smooth", "hysteresis", "merge_intervals"]

LOG = logging.getLogger("worker.events.detectors")

Series = list[Optional[float]]

_DEFAULTS: dict[str, float] = {
    "p_on": 0.6,
    "p_off": 0.4,
    "min_duration_sec": 1.0,
    "merge_gap_sec": 5.0,
    "smooth_points": 3,
}


@dataclass(frozen=True)
class Interval:
    """Интервал нарушения по индексам состояний [start, end] включительно."""

    start: int
    end: int


# --------------------------------------------------------------------------- общая механика


def _param(params: Mapping[str, Any], name: str, default: Optional[float] = None) -> float:
    value = params.get(name, _DEFAULTS.get(name) if default is None else default)
    if value is None:
        raise ValueError(f"нет параметра {name}")
    return float(value)


def smooth(values: Sequence[Optional[float]], points: int) -> Series:
    """Скользящая медиана по окну points (по центру); None в окне пропускаются."""
    points = max(1, int(points))
    half = points // 2
    out: Series = []
    for i in range(len(values)):
        window = [v for v in values[max(0, i - half): i + half + 1] if v is not None]
        out.append(median(window) if window and values[i] is not None else None)
    return out


def hysteresis(values: Sequence[Optional[float]], p_on: float, p_off: float) -> list[Interval]:
    """Интервалы, где ряд «включён»: вход при p >= p_on, выход при p < p_off; None — без изменений."""
    intervals: list[Interval] = []
    start: Optional[int] = None
    last_on: Optional[int] = None
    for i, v in enumerate(values):
        if v is None:
            continue
        if start is None:
            if v >= p_on:
                start = last_on = i
        elif v < p_off:
            intervals.append(Interval(start, last_on if last_on is not None else start))
            start = last_on = None
        else:
            last_on = i
    if start is not None:
        intervals.append(Interval(start, last_on if last_on is not None else start))
    return intervals


def merge_intervals(intervals: Sequence[Interval], times: Sequence[float], gap_sec: float) -> list[Interval]:
    """Склеить интервалы, между которыми меньше gap_sec."""
    merged: list[Interval] = []
    for iv in intervals:
        if merged and times[iv.start] - times[merged[-1].end] <= gap_sec:
            merged[-1] = Interval(merged[-1].start, iv.end)
        else:
            merged.append(iv)
    return merged


def _evidence(scene: SceneTimeline, iv: Interval, values: Series) -> tuple[list[float], list[str]]:
    """Кадры-доказательства: начало, +1 с, максимум p. Возвращает таймкоды и пути кадров."""
    states = scene.states
    t0 = states[iv.start].t
    candidates = [iv.start]
    plus_one = next((i for i in range(iv.start, len(states)) if states[i].t >= t0 + 1.0 - 1e-6), None)
    if plus_one is not None:
        candidates.append(plus_one)
    peak = max(range(iv.start, iv.end + 1), key=lambda i: values[i] if values[i] is not None else -1.0)
    candidates.append(peak)
    idx = list(dict.fromkeys(candidates))[:3]
    return [round(states[i].t, 3) for i in idx], [states[i].frame for i in idx]


def _events_from_series(
    scene: SceneTimeline,
    raw: Series,
    params: Mapping[str, Any],
    describe: Callable[[SceneTimeline, Interval], dict[str, Any]],
) -> list[Event]:
    """Общий конвейер: медиана -> гистерезис -> склейка -> длительность -> события."""
    if not scene.states:
        return []
    values = smooth(raw, int(_param(params, "smooth_points")))
    times = [s.t for s in scene.states]
    intervals = hysteresis(values, _param(params, "p_on"), _param(params, "p_off"))
    intervals = merge_intervals(intervals, times, _param(params, "merge_gap_sec"))
    step = 1.0 / scene.fps_sampled
    min_dur = _param(params, "min_duration_sec")
    events: list[Event] = []
    for iv in intervals:
        if times[iv.end] - times[iv.start] + step < min_dur:
            continue
        on = [v for v in values[iv.start: iv.end + 1] if v is not None]
        stamps, frames = _evidence(scene, iv, values)
        facts = describe(scene, iv)
        facts["frames"] = frames
        events.append(Event(
            rule_id="",
            timecode_sec=round(times[iv.start], 1),
            confidence=round(mean(on), 3) if on else 0.0,
            evidence_timestamps_sec=stamps,
            facts=facts,
        ))
    return events


def _fmin(*values: Optional[float]) -> Optional[float]:
    """Логическое «и» для вероятностей: минимум; нет данных хоть по одному — None."""
    return None if any(v is None for v in values) else min(values)  # type: ignore[type-var]


def _fmax(*values: Optional[float]) -> Optional[float]:
    """Логическое «или»: максимум известных; все неизвестны — None."""
    known = [v for v in values if v is not None]
    return max(known) if known else None


def _away(state: SceneState) -> Optional[float]:
    """Вероятность, что кассира нет на рабочем месте."""
    return None if state.cashier_present is None else 1.0 - state.cashier_present


def _overlap(a: Optional[BBox], b: Optional[BBox]) -> float:
    """Доля площади a, лежащая внутри b (0, если рамок нет)."""
    if a is None or b is None:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if area > 0 else 0.0


def _expand(box: Optional[BBox], margin: float) -> Optional[BBox]:
    if box is None:
        return None
    return (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)


def _peak(scene: SceneTimeline, iv: Interval, name: str) -> Optional[float]:
    vals = [scene.states[i].value(name) for i in range(iv.start, iv.end + 1)]
    known = [v for v in vals if v is not None]
    return max(known) if known else None


def _present(scene: SceneTimeline, iv: Interval, name: str, threshold: float = 0.5) -> bool:
    p = _peak(scene, iv, name)
    return p is not None and p >= threshold


def _zone(scene: SceneTimeline, name: str) -> Optional[list[float]]:
    box = scene.zones.get(name)
    return list(box) if box is not None else None


# --------------------------------------------------------------------------- детекторы


def personal_items(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R01: верхняя одежда (и, по params.include_bag, личная сумка) на рабочем столе."""
    include_bag = bool(params.get("include_bag", False))
    raw = [_fmax(s.outerwear_on_desk, s.bag_on_desk if include_bag else None) for s in scene.states]

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        items = ["верхняя одежда"] if _present(sc, iv, "outerwear_on_desk") else []
        if include_bag and _present(sc, iv, "bag_on_desk"):   # без include_bag сумка не проверялась
            items.append("сумка")
        return {"what_visible": ", ".join(items) or "предмет одежды", "bbox": _zone(sc, "desk")}

    return _events_from_series(scene, raw, params, describe)


def storage_left_open(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R02: сейф открыт, а кассир не у сейфа (рамка кассира не пересекается с зоной сейфа)."""
    safe_zone = _expand(scene.zones.get("safe"), float(params.get("safe_margin", 0.05)))
    near_share = float(params.get("near_safe_overlap", 0.15))
    raw: Series = []
    for s in scene.states:
        if safe_zone is None:
            away: Optional[float] = None
        elif s.cashier_bbox is None:
            away = 1.0 if s.cashier_present is not None else None
        else:
            away = 0.0 if _overlap(s.cashier_bbox, safe_zone) >= near_share else 1.0
        raw.append(_fmin(s.safe_open, away))

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        where = "кассир за рабочим столом" if _present(sc, iv, "cashier_present") else "кассира нет у рабочего места"
        return {"what_visible": f"дверца сейфа открыта, {where}", "bbox": _zone(sc, "safe")}

    return _events_from_series(scene, raw, params, describe)


def valuables_unattended(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R03: кассира нет на рабочем месте, на столе наличные или сумка."""
    raw = [_fmin(_away(s), _fmax(s.cash_on_desk, s.bag_on_desk)) for s in scene.states]

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        items = []
        if _present(sc, iv, "cash_on_desk"):
            items.append("наличные")
        if _present(sc, iv, "bag_on_desk"):
            items.append("сумка")
        return {"what_visible": " и ".join(items) or "ценности", "bbox": _zone(sc, "desk")}

    return _events_from_series(scene, raw, params, describe)


def outsider_in_cash_room(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R04: вход человека в кассу, когда (а) в комнате становится больше max_persons_normal,
    (б) кассир в этот момент работает с наличными или (в) вход сразу после выхода другого человека.

    Момент входа — person_entered или first_seen_t нового трека в others. Вошедший должен
    оставаться в комнате не меньше min_duration_sec.
    """
    states = scene.states
    if not states:
        return []
    max_normal = int(params.get("max_persons_normal", 2))
    cash_p = float(params.get("cash_p", 0.5))
    after_exit = float(params.get("after_exit_sec", 5.0))
    min_dur = _param(params, "min_duration_sec")
    gap = _param(params, "merge_gap_sec")

    entries: list[float] = [s.t for s in states if s.person_entered]
    for s in states:
        entries.extend(p.first_seen_t for p in s.others if p.first_seen_t > states[0].t + 0.5)
    entries.sort()
    moments: list[float] = []
    for t in entries:
        if not moments or t - moments[-1] > gap:
            moments.append(t)
    exits = [s.t for s in states if s.person_exited]
    cash = smooth([s.cash_on_desk for s in states], int(_param(params, "smooth_points")))

    events: list[Event] = []
    for t in moments:
        before = [i for i, s in enumerate(states) if t - 2.0 <= s.t < t]
        after = [i for i, s in enumerate(states) if t <= s.t <= t + max(min_dur, 1.0)]
        if not after:
            continue
        counts = [states[i].persons_in_room for i in after if states[i].persons_in_room is not None]
        if not counts or min(counts) < 2:
            continue                                   # вошедший не задержался в кассе
        reasons: list[tuple[float, str]] = []
        if max(counts) > max_normal:
            reasons.append((0.85, f"в кассе {max(counts)} человека"))
        cash_before = [cash[i] for i in before if cash[i] is not None]
        if cash_before and max(cash_before) >= cash_p:
            reasons.append((max(cash_before), "на столе наличные, с которыми работает кассир"))
        if any(0.0 < t - e <= after_exit for e in exits):
            reasons.append((0.6, "вход сразу после выхода предыдущего посетителя"))
        if not reasons:
            continue
        conf = max(r[0] for r in reasons)
        i0 = after[0]
        idx = list(dict.fromkeys([i0, *(i for i in after if states[i].t >= t + 1.0)][:2]))
        events.append(Event(
            rule_id="",
            timecode_sec=round(t, 1),
            confidence=round(conf, 3),
            evidence_timestamps_sec=[round(states[i].t, 3) for i in idx],
            facts={
                "what_visible": "; ".join(r[1] for r in reasons),
                "bbox": _zone(scene, "door"),
                "frames": [states[i].frame for i in idx],
            },
        ))
    return events


def collection_inside(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R05: инкассаторская сумка в кассе, когда в комнате кроме кассира есть другой человек."""
    raw = [_fmin(s.collection_bag, 1.0 if s.others else 0.0) for s in scene.states]

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        return {"what_visible": "в кассе вошедший человек и инкассаторская сумка", "bbox": _zone(sc, "desk")}

    return _events_from_series(scene, raw, params, describe)


def drink_at_workplace(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R06: стакан или еда у кассира на рабочем месте."""
    raw = [_fmin(s.cup_or_food, s.cashier_present) for s in scene.states]

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        tail = ", рядом наличные" if _present(sc, iv, "cash_on_desk") else ""
        return {"what_visible": f"стакан или еда у кассира{tail}", "bbox": None}

    return _events_from_series(scene, raw, params, describe)


def phone_in_cash_room(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R07: телефон в руке кассира."""
    raw = [_fmin(s.phone_in_hand, s.cashier_present) for s in scene.states]

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        return {"what_visible": "телефон в руке кассира", "bbox": None}

    return _events_from_series(scene, raw, params, describe)


def left_during_service(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """R08: в кассе посетитель, кассира нет на рабочем месте."""
    raw = [_fmin(_away(s), (1.0 if s.others else 0.0) if s.persons_in_room is not None else None)
           for s in scene.states]

    def describe(sc: SceneTimeline, iv: Interval) -> dict[str, Any]:
        return {"what_visible": "посетитель в кассе, кассира нет на рабочем месте", "bbox": _zone(sc, "workplace")}

    return _events_from_series(scene, raw, params, describe)


DETECTORS: dict[str, Callable[[SceneTimeline, Mapping[str, Any]], list[Event]]] = {
    "personal_items": personal_items,
    "storage_left_open": storage_left_open,
    "valuables_unattended": valuables_unattended,
    "outsider_in_cash_room": outsider_in_cash_room,
    "collection_inside": collection_inside,
    "drink_at_workplace": drink_at_workplace,
    "phone_in_cash_room": phone_in_cash_room,
    "left_during_service": left_during_service,
}

# Детекторы позы — в отдельном модуле; импорт после всех определений, чтобы модуль позы мог
# брать общую механику (_events_from_series) без циклического импорта.
from worker.events.pose_detectors import POSE_DETECTORS  # noqa: E402

DETECTORS.update(POSE_DETECTORS)
