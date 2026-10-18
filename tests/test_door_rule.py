"""R15 «Дверь кассы оставлена открытой» (worker.events.detectors.door_left_open) на синтетических сценах
с параметрами из rules.yaml: позитив как Х-32 и Х-33 и трудные негативы — дверь закрыли после прохода,
погас свет (Х-11, Х-12), изменение без прохода, два прохода подряд, кэш без поля door_change."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pytest

from worker.events import detectors as det
from worker.scene.contract import Person, SceneState, SceneTimeline

FPS = 3.0
DOOR = (0.0, 0.05, 0.25, 0.5)
AT_DOOR = (0.02, 0.1, 0.22, 0.55)        # человек в проёме
SEATED = (0.4, 0.4, 0.7, 0.85)           # кассир за столом, далеко от двери


@pytest.fixture(scope="module")
def r15(rules):
    return next(r for r in rules if r.id == "R15_door_left_open")


def _scene(seconds: float, state: Callable[[float], dict[str, Any]], door: Optional[tuple] = DOOR) -> SceneTimeline:
    states = []
    for i in range(int(seconds * FPS)):
        t = round(i / FPS, 3)
        fields: dict[str, Any] = {"persons_in_room": 1, "cashier_present": 1.0, "cashier_bbox": SEATED,
                                  "brightness": 0.54, "door_change": 0.0}
        fields.update(state(t))
        others = tuple(fields.pop("others", ()))
        known = {k: fields.pop(k) for k in list(fields) if k in SceneState.__dataclass_fields__}
        states.append(SceneState(t=t, frame=f"frames/{i:06d}.jpg", others=others, extra=fields, **known))
    return SceneTimeline(video_id="Х-синт", duration_sec=seconds, fps_sampled=FPS, frame_size=(464, 848),
                         zones={"desk": (0.05, 0.4, 0.7, 0.65), "safe": None, "door": door, "workplace": None,
                                "laptop": None},
                         states=tuple(states))


def _enter(t: float, start: float = 1.5, end: float = 4.5) -> dict[str, Any]:
    """Кассир входит: у двери с start до end, вход засчитан на первом кадре у двери."""
    if start <= t < end:
        return {"cashier_bbox": AT_DOOR, "cashier_present": 0.0, "cashier_returned": abs(t - start) < 0.2}
    return {}


def test_registered_and_enabled(r15):
    assert det.DETECTORS["door_left_open"] is det.door_left_open
    assert r15.detector == "door_left_open"
    assert r15.source.url == "https://adilet.zan.kz/rus/docs/V2000020075"


def test_door_left_open_after_entry(r15):
    """Х-32: кассир входит, дверь остаётся открытой до конца ролика."""
    scene = _scene(20, lambda t: {**_enter(t), "door_change": 1.0 if t >= 3.0 else 0.0})
    events = det.door_left_open(scene, r15.params)
    assert len(events) == 1
    ev = events[0]
    assert 4.3 <= ev.timecode_sec <= 5.3                      # кассир отошёл от двери; эталон Х-32 — 5 с
    assert ev.confidence >= 0.9
    assert "открытой" in ev.facts["what_visible"]
    assert ev.facts["bbox"] == list(DOOR)
    assert "входит человек" not in ev.facts["what_visible"]


def test_visitor_walks_in_through_open_door(r15):
    """Х-33: в открытую дверь на 10 с входит человек — это в описании (и отдельно R11)."""
    def state(t: float) -> dict[str, Any]:
        s = {**_enter(t), "door_change": 1.0 if t >= 3.0 else 0.0}
        if abs(t - 10.0) < 0.2:
            s["visitor_entered_ids"] = [7]
        return s

    events = det.door_left_open(_scene(15, state), r15.params)
    assert len(events) == 1
    assert "входит человек" in events[0].facts["what_visible"]


def test_door_closed_after_passage(r15):
    """Вход с закрытием двери (Х-3, Х-35): вид двери меняется только на время прохода."""
    scene = _scene(20, lambda t: {**_enter(t, 2.0, 4.0), "door_change": 1.0 if 2.0 <= t < 5.0 else 0.05})
    assert det.door_left_open(scene, r15.params) == []


def test_lights_off_when_cashier_leaves(r15):
    """Х-11, Х-12: кассир уходит, свет гаснет — вид двери другой до конца ролика, но это свет."""
    def state(t: float) -> dict[str, Any]:
        s = _enter(t, 10.0, 12.0)
        s["door_change"] = 1.0 if t >= 12.0 else 0.0
        s["brightness"] = 0.36 if t >= 12.0 else 0.54
        return s

    assert det.door_left_open(_scene(30, state), r15.params) == []


def test_change_without_passage(r15):
    """Сдвиг камеры или отражение: вид двери меняется, а через дверь никто не проходил."""
    scene = _scene(30, lambda t: {"door_change": 1.0 if t >= 10.0 else 0.0})
    assert det.door_left_open(scene, r15.params) == []


def test_two_quick_passages_are_not_merged(r15):
    """Уход клиента и вход следующего (Х-5, Х-19): два коротких изменения подряд — не одно событие."""
    def state(t: float) -> dict[str, Any]:
        at_door = 10.0 <= t < 11.0 or 14.5 <= t < 15.5
        client = Person(track_id=3, bbox=AT_DOOR, first_seen_t=0.0) if at_door else None
        return {"others": (client,) if client else (), "persons_in_room": 2 if client else 1,
                "door_change": 1.0 if 10.0 <= t < 13.0 or 14.5 <= t < 17.5 else 0.05}

    scene = _scene(30, state)
    assert det.door_left_open(scene, r15.params) == []
    merged = {**r15.params, "merge_gap_sec": 5.0}            # с прежней склейкой два прохода дали бы событие
    assert len(det.door_left_open(scene, merged)) == 1


def test_person_standing_in_doorway_is_not_scored(r15):
    """Человек весь интервал стоит в проёме: дверь держат открытой при нём — оценки нет."""
    scene = _scene(20, lambda t: {"others": (Person(track_id=5, bbox=AT_DOOR, first_seen_t=0.0),),
                                  "persons_in_room": 2, "door_change": 1.0 if t >= 3.0 else 0.0})
    assert det.door_left_open(scene, r15.params) == []


def test_silent_without_door_zone_or_signal(r15):
    opened = lambda t: {**_enter(t), "door_change": 1.0 if t >= 3.0 else 0.0}  # noqa: E731
    assert det.door_left_open(_scene(20, opened, door=None), r15.params) == []
    old_cache = lambda t: {**_enter(t), "door_change": None}  # noqa: E731  (кэш собран до R15)
    assert det.door_left_open(_scene(20, old_cache), r15.params) == []


def test_door_not_latched_after_visitor_is_not_violation(r15):
    """Х-39: посетитель вошёл, створка прикрыта неплотно (отражения в стекле сдвинуты) — door_diff держится
    0,55–0,58, то есть door_change 0,67–0,76, 20 с. Не нарушение (решение по разметке 18.10): p_on 0,9."""
    scene = _scene(30, lambda t: {**_enter(t), "door_change": 0.75 if t >= 5.0 else 0.0})
    assert det.door_left_open(scene, r15.params) == []


def test_weakest_positive_still_fires(r15):
    """Х-32: удерживаемый 6 с уровень door_diff 0,71 (door_change 1,0 после обрезки), плато не ниже 0,62."""
    scene = _scene(20, lambda t: {**_enter(t), "door_change": 0.95 if t >= 3.0 else 0.0})
    assert len(det.door_left_open(scene, r15.params)) == 1


def test_timecode_when_entrant_left_the_door(r15):
    """Х-33: дверь распахнулась на 3,3 с, детектор теряет кассира в проёме; она снова видна у двери на 4,0
    и 6,3 с и отходит к 6,7 с. Таймкод — когда отошла (эталон 6 с), а не когда дверь распахнулась."""
    def state(t: float) -> dict[str, Any]:
        st: dict[str, Any] = {"door_change": 1.0 if t >= 3.3 else 0.0}
        if 1.3 <= t < 2.6 or abs(t - 4.0) < 0.1 or abs(t - 6.33) < 0.1:
            st.update(cashier_bbox=AT_DOOR, cashier_present=0.0, cashier_returned=abs(t - 1.33) < 0.1)
        elif 2.6 <= t < 6.6:
            st.update(cashier_bbox=None, cashier_present=0.0, persons_in_room=0)
        return st
    events = det.door_left_open(_scene(20, state), r15.params)
    assert len(events) == 1
    assert 6.0 <= events[0].timecode_sec <= 7.0
    old = det.door_left_open(_scene(20, state), {**r15.params, "settle_sec": 0.0})
    assert old[0].timecode_sec < 4.0                                 # без settle_sec — момент распахивания


def _crowded_doorway(t: float) -> dict[str, Any]:
    """Дверь открыта с 3 до конца, человек в проёме каждые 1,33 с до 12 с — спокойного окна нет."""
    st: dict[str, Any] = {**_enter(t), "door_change": 1.0 if t >= 3.0 else 0.0}
    if 3.0 <= t <= 12.0 and round(t * 3) % 4 == 0:
        st.update(cashier_bbox=AT_DOOR, cashier_present=0.0)
    return st


def test_settle_not_found_keeps_event_at_interval_start(r15):
    """Ревью 18.10: без спокойного окна за settle_max_sec событие остаётся с таймкодом начала, а не пропадает."""
    events = det.door_left_open(_scene(25, _crowded_doorway), r15.params)
    base = det.door_left_open(_scene(25, _crowded_doorway), {**r15.params, "settle_sec": 0.0})
    assert len(events) == len(base) == 1
    assert events[0].timecode_sec == base[0].timecode_sec


def test_settle_keeps_first_frame_and_evidence_on_timecode(r15):
    """Дверь открылась на 4,667 с (таймкод округлён до 4,7), у двери после этого никого: таймкод не съезжает
    на кадр; при сдвиге первый кадр-доказательство — кадр таймкода."""
    scene = _scene(20, lambda t: {**_enter(t, 1.5, 4.6), "door_change": 1.0 if t >= 4.6 else 0.0})
    plain = det.door_left_open(scene, {**r15.params, "settle_sec": 0.0})
    events = det.door_left_open(scene, r15.params)
    assert [e.timecode_sec for e in events] == [e.timecode_sec for e in plain]
    ev = det.door_left_open(_scene(20, lambda t: {
        **_enter(t), "door_change": 1.0 if t >= 3.3 else 0.0,
        **({"cashier_bbox": AT_DOOR, "cashier_present": 0.0} if abs(t - 6.33) < 0.1 else {})}), r15.params)[0]
    assert ev.evidence_timestamps_sec[0] == pytest.approx(ev.timecode_sec, abs=0.05)


def test_person_lingering_at_open_door_is_not_violation(r15):
    """crop Х-16: человек ~15 с стоит у открытой двери (кадры «у двери» пропускаются, интервал не рвут) — оценённых
    кадров по краям меньше min_scored_sec, события нет."""
    def state(t: float) -> dict[str, Any]:
        st: dict[str, Any] = {**_enter(t, 1.5, 2.6), "door_change": 1.0 if t >= 3.0 else 0.0, "door_diff": 0.8}
        if 3.6 <= t < 18.0:
            st.update(cashier_bbox=AT_DOOR, cashier_present=0.0)
        elif t >= 19.0:
            st.update(door_change=0.0)
        return st
    assert det.door_left_open(_scene(30, state), r15.params) == []
    assert det.door_left_open(_scene(30, state), {**r15.params, "min_scored_sec": 0.0})     # без фильтра — ложное


def test_door_pulled_back_after_passage_is_not_violation(r15):
    """Х-39: при проходе door_diff 0,95, потом створку прикрыли не до конца — плато 0,62 (door_change 0,9 выше p_on),
    0,62 / 0,95 = 0,65 < min_held_ratio; дверь, оставшаяся открытой так же, — событие."""
    def ajar(t: float) -> dict[str, Any]:
        d = 0.95 if 3.0 <= t < 5.0 else 0.62 if t >= 5.0 else 0.03
        return {**_enter(t), "door_diff": d, "door_change": min(1.0, max(0.0, (d - 0.35) / 0.3))}
    assert det.door_left_open(_scene(30, ajar), r15.params) == []

    def open_(t: float) -> dict[str, Any]:
        d = 0.75 if t >= 3.0 else 0.03
        return {**_enter(t), "door_diff": d, "door_change": min(1.0, max(0.0, (d - 0.35) / 0.3))}
    assert len(det.door_left_open(_scene(30, open_), r15.params)) == 1
