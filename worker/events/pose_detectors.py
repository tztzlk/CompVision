"""Детекторы по признакам позы (поля состояний из worker.scene.pose_pass).

Реестр POSE_DETECTORS подключается в конце worker/events/detectors.py:
DETECTORS.update(POSE_DETECTORS). Правила с этими детекторами пока только в блоке
«ЗАПЛАНИРОВАНО» rules.yaml: без сверенной нормы и замера по кэшу они не включаются.

Общая механика — detectors._events_from_series: таймкод — начало жеста, confidence — среднее
значение признака, кадры — начало, +1 с и пик. Без прохода позы (ENABLE_POSE не задан) поля
равны None и детекторы событий не дают.

Общая механика импортируется внутри функций: detectors.py импортирует этот модуль в своём
конце, а прямой импорт pose_detectors раньше detectors иначе дал бы цикл.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from worker.events.model import Event
from worker.scene.contract import SceneTimeline

__all__ = [
    "POSE_DETECTORS",
    "storage_left_open_pose",
    "hand_to_hand_transfer",
    "counting_without_controller",
    "drink_or_phone_at_face",
]


def _det() -> Any:
    from worker.events import detectors

    return detectors


def _alone(persons: Optional[int]) -> Optional[float]:
    """1 — кассир в комнате один, 0 — есть ещё люди, None — число людей неизвестно."""
    if persons is None:
        return None
    return 1.0 if persons <= 1 else 0.0


def hand_to_hand_transfer(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """Передача из рук в руки: кисть кассира рядом с кистью другого человека над столом.

    params.min_persons (по умолчанию 2) — сколько людей должно быть в комнате; 3 — передача
    при третьем человеке (как в Х-18), если обычная передача клиенту окажется нормой.
    """
    d = _det()
    min_persons = int(params.get("min_persons", 2))
    raw = []
    for s in scene.states:
        crowd = None if s.persons_in_room is None else (1.0 if s.persons_in_room >= min_persons else 0.0)
        raw.append(d._fmin(s.value("hand_to_hand"), crowd))

    def describe(sc: SceneTimeline, iv: Any) -> dict[str, Any]:
        tracks = [sc.states[i].value("hand_to_hand_track") for i in range(iv.start, iv.end + 1)]
        known = [t for t in tracks if isinstance(t, int)]
        who = f" (человек #{max(set(known), key=known.count)})" if known else ""
        return {"what_visible": f"кисть кассира рядом с кистью другого человека{who} над столом",
                "bbox": d._zone(sc, "desk")}

    return d._events_from_series(scene, raw, {"min_duration_sec": 0.5, **params}, describe)


def counting_without_controller(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """Пересчёт руками в одиночку: кисти вместе над столом и двигаются, в комнате больше никого."""
    d = _det()
    raw = [d._fmin(s.value("cashier_counting"), _alone(s.persons_in_room)) for s in scene.states]

    def describe(sc: SceneTimeline, iv: Any) -> dict[str, Any]:
        cash = ", на столе наличные" if d._present(sc, iv, "cash_on_desk") else ""
        return {"what_visible": f"кассир перебирает что-то обеими руками над столом{cash}; других людей в кассе не видно",
                "bbox": d._zone(sc, "desk")}

    return d._events_from_series(scene, raw, {"min_duration_sec": 5.0, **params}, describe)


def drink_or_phone_at_face(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """Рука у лица, и в кадре стакан/еда или телефон в руке.

    Предмет — максимум полей params.object_fields: по умолчанию desk_cup (стакан на столе по скану
    OWLv2 — на нём стоит R13; на Х-46 COCO cup_or_food после медианы не доходит до p_on), cup_or_food
    и phone_in_hand (COCO у рук).
    """
    d = _det()
    fields = tuple(params.get("object_fields", ("desk_cup", "cup_or_food", "phone_in_hand")))
    raw = [d._fmin(s.value("cashier_hand_at_face"), d._fmax(*(s.value(f) for f in fields)))
           for s in scene.states]

    def describe(sc: SceneTimeline, iv: Any) -> dict[str, Any]:
        cup = d._fmax(d._peak(sc, iv, "cup_or_food"), d._peak(sc, iv, "desk_cup"))
        phone = d._peak(sc, iv, "phone_in_hand")
        item = "телефон" if (phone or 0.0) > (cup or 0.0) else "стакан или еда"
        return {"what_visible": f"кассир подносит руку к лицу, в кадре {item}", "bbox": None}

    return d._events_from_series(scene, raw, params, describe)


def storage_left_open_pose(scene: SceneTimeline, params: Mapping[str, Any]) -> list[Event]:
    """Улучшенный R10: кассир работал руками в сейфе, отошёл, а сейф остался открытым.

    1. Работа с сейфом — cashier_hand_in_safe (медиана, гистерезис p_on/p_off) не короче
       min_work_sec (0,7 с).
    2. Момент ухода — первое состояние после работы, где кисти нет в сейфе и рамка кассира
       не заходит в зону сейфа (доля < near_safe_overlap).
    3. Нарушение — с момента ухода сейф открыт (safe_open, или safe_changed, если его посчитал
       отдельный проход) не короче min_duration_sec в пределах open_window_sec, и кассир за это
       время к сейфу не вернулся. Таймкод — момент ухода.
    """
    d = _det()
    states = scene.states
    if not states:
        return []
    p_on, p_off = d._param(params, "p_on"), d._param(params, "p_off")
    points = int(d._param(params, "smooth_points"))
    min_work = float(params.get("min_work_sec", 0.7))
    min_dur = d._param(params, "min_duration_sec")
    # Окно не короче минимальной длительности: с параметрами R10 (min_duration_sec 7) окно 4 с
    # не давало событий никогда.
    window = max(float(params.get("open_window_sec", 4.0)), min_dur + 1.0 / scene.fps_sampled)
    gap = d._param(params, "merge_gap_sec")
    near_share = float(params.get("near_safe_overlap", 0.15))
    safe_zone = d._expand(scene.zones.get("safe"), float(params.get("safe_margin", 0.05)))
    if safe_zone is None:
        return []
    times = [s.t for s in states]
    step = 1.0 / scene.fps_sampled
    hand = d.smooth([s.value("cashier_hand_in_safe") for s in states], points)
    opened = d.smooth([d._fmax(s.safe_open, s.value("safe_changed")) for s in states], points)

    def at_safe(i: int) -> bool:
        h = hand[i]
        return (h is not None and h >= 0.5) or d._overlap(states[i].cashier_bbox, safe_zone) >= near_share

    events: list[Event] = []
    for work in d.hysteresis(hand, p_on, p_off):
        if times[work.end] - times[work.start] + step < min_work:
            continue
        leave = next((i for i in range(work.end + 1, len(states)) if not at_safe(i)), None)
        if leave is None or (events and times[leave] - events[-1].timecode_sec <= gap):
            continue
        span = [i for i in range(leave, len(states)) if times[i] <= times[leave] + window]
        on = [i for i in span if opened[i] is not None and opened[i] >= p_on and not at_safe(i)]
        if not on or times[on[-1]] - times[on[0]] + step < min_dur:
            continue
        if any(at_safe(i) for i in range(leave, on[-1] + 1)):
            continue                              # вернулся к сейфу — это ещё работа с ним
        values = [opened[i] for i in on]
        peak = max(on, key=lambda i: opened[i] or 0.0)
        plus_one = next((i for i in span if times[i] >= times[leave] + 1.0 - 1e-6), None)
        idx = list(dict.fromkeys([leave, *([plus_one] if plus_one is not None else []), peak]))[:3]
        events.append(Event(
            rule_id="",
            timecode_sec=round(times[leave], 1),
            confidence=round(sum(values) / len(values), 3),  # type: ignore[arg-type]
            evidence_timestamps_sec=[round(times[i], 3) for i in idx],
            facts={
                "what_visible": "кассир убрал руки из сейфа и отошёл, дверца сейфа остаётся открытой",
                "bbox": d._zone(scene, "safe"),
                "frames": [states[i].frame for i in idx],
            },
        ))
    return events


POSE_DETECTORS: dict[str, Callable[[SceneTimeline, Mapping[str, Any]], list[Event]]] = {
    "storage_left_open_pose": storage_left_open_pose,
    "hand_to_hand_transfer": hand_to_hand_transfer,
    "counting_without_controller": counting_without_controller,
    "drink_or_phone_at_face": drink_or_phone_at_face,
}
