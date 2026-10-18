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
