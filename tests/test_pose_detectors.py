"""Детекторы позы (worker/events/pose_detectors.py) на синтетических сценах: позитив и трудный негатив.

Признаки позы задаются прямо в полях состояний (как их пишет worker.scene.pose_pass).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from worker.events import detectors as det
from worker.events.pose_detectors import (
    POSE_DETECTORS,
    counting_without_controller,
    drink_or_phone_at_face,
    hand_to_hand_transfer,
    storage_left_open_pose,
)
from worker.scene.contract import Person, SceneState, SceneTimeline, load_scene

FPS = 3.0
SAFE = (0.62, 0.28, 0.82, 0.55)
DESK = (0.05, 0.38, 0.62, 0.62)
SEATED_BOX = (0.3, 0.4, 0.6, 0.85)          # у стола, в зону сейфа не заходит
AT_SAFE_BOX = (0.55, 0.3, 0.8, 0.8)
FIXTURES = Path(__file__).parent / "fixtures"


def _scene(seconds: float, state: Callable[[float], dict[str, Any]]) -> SceneTimeline:
    states = []
    for i in range(int(seconds * FPS)):
        t = round(i / FPS, 3)
        fields = {"persons_in_room": 1, "cashier_present": 0.95, "cashier_bbox": SEATED_BOX}
        fields.update(state(t))
        others = tuple(Person(track_id=4, bbox=(0.35, 0.2, 0.55, 0.45), first_seen_t=0.0)
                       for _ in range(max(0, fields["persons_in_room"] - 1)))[:1]
        known = {k: fields.pop(k) for k in list(fields) if k in SceneState.__dataclass_fields__}
        states.append(SceneState(t=t, frame=f"frames/{i:06d}.jpg", others=others, extra=fields, **known))
    return SceneTimeline(video_id="Х-синт", duration_sec=seconds, fps_sampled=FPS, frame_size=(464, 848),
                         zones={"desk": DESK, "safe": SAFE, "door": None, "workplace": None, "laptop": None},
                         states=tuple(states))


def _between(t: float, a: float, b: float, inside: Any, outside: Any = 0.0) -> Any:
    return inside if a <= t < b else outside


def test_registry_and_no_events_without_pose_fields():
    assert set(POSE_DETECTORS) <= set(det.DETECTORS)
    for name in ("X15", "X18", "X20", "X23", "X40"):         # сцены без полей позы: None -> событий нет
        scene = load_scene(FIXTURES / f"scene_{name}.json")
        for fn in POSE_DETECTORS.values():
            assert fn(scene, {}) == []


# ---------------------------------------------------------------- передача из рук в руки


def _h2h(t: float, start: float = 4.0, end: float = 6.0, persons: int = 2) -> dict[str, Any]:
    return {"persons_in_room": persons, "hand_to_hand": _between(t, start, end, 0.9),
            "hand_to_hand_track": _between(t, start, end, 4, None)}


def test_hand_to_hand_positive():
    events = hand_to_hand_transfer(_scene(10, _h2h), {})
    assert len(events) == 1
    assert abs(events[0].timecode_sec - 4.0) <= 0.4
    assert events[0].confidence >= 0.8
    assert "#4" in events[0].facts["what_visible"]


def test_hand_to_hand_hard_negatives():
    brief = _scene(10, lambda t: _h2h(t, 4.0, 4.2))                   # касание на одном кадре
    assert hand_to_hand_transfer(brief, {}) == []
    # обычная передача клиенту, а правило требует третьего человека в комнате
    assert hand_to_hand_transfer(_scene(10, _h2h), {"min_persons": 3}) == []
    assert len(hand_to_hand_transfer(_scene(10, lambda t: _h2h(t, persons=3)), {"min_persons": 3})) == 1
    unknown = _scene(10, lambda t: {"persons_in_room": 2, "hand_to_hand": None})   # кисти не видны
    assert hand_to_hand_transfer(unknown, {}) == []


# ---------------------------------------------------------------- пересчёт без контролёра


def test_counting_alone_positive():
    scene = _scene(15, lambda t: {"cashier_counting": _between(t, 3.0, 10.0, 0.8), "cash_on_desk": 0.7})
    events = counting_without_controller(scene, {})
    assert len(events) == 1
    assert abs(events[0].timecode_sec - 3.0) <= 0.4
    assert "наличные" in events[0].facts["what_visible"]


def test_counting_hard_negatives():
    with_client = _scene(15, lambda t: {"persons_in_room": 2, "cashier_counting": _between(t, 3.0, 10.0, 0.8)})
    assert counting_without_controller(with_client, {}) == []          # пересчёт при клиенте — норма
    short = _scene(15, lambda t: {"cashier_counting": _between(t, 3.0, 6.0, 0.8)})
    assert counting_without_controller(short, {}) == []                # 3 с < 5 с


# ---------------------------------------------------------------- рука у лица


def test_drink_at_face_positive():
    scene = _scene(15, lambda t: {"cashier_hand_at_face": _between(t, 11.0, 13.0, 0.9), "cup_or_food": 0.8,
                                  "phone_in_hand": 0.0})
    events = drink_or_phone_at_face(scene, {})
    assert len(events) == 1
    assert abs(events[0].timecode_sec - 11.0) <= 0.4
    assert "стакан" in events[0].facts["what_visible"]


def test_hand_at_face_without_object_is_not_event():
    scene = _scene(15, lambda t: {"cashier_hand_at_face": _between(t, 11.0, 13.0, 0.9), "cup_or_food": 0.05,
                                  "phone_in_hand": 0.0})
    assert drink_or_phone_at_face(scene, {}) == []                     # почесал лицо


# ---------------------------------------------------------------- сейф (улучшенный R10)


def _safe(t: float, work=(2.0, 3.5), open_until: Optional[float] = 15.0) -> dict[str, Any]:
    working = work[0] <= t < work[1]
    return {
        "cashier_hand_in_safe": 1.0 if working else 0.0,
        "cashier_bbox": AT_SAFE_BOX if working else SEATED_BOX,
        "safe_open": 0.9 if 1.5 <= t < (open_until or 0.0) else 0.05,
    }


def test_safe_left_open_after_work_positive():
    events = storage_left_open_pose(_scene(15, _safe), {})
    assert len(events) == 1
    assert abs(events[0].timecode_sec - 3.5) <= 0.4                     # руки ушли из сейфа
    assert events[0].confidence >= 0.8
    assert events[0].facts["bbox"] == list(SAFE)


def test_safe_hard_negatives():
    closed = _scene(15, lambda t: _safe(t, open_until=4.0))            # закрыл сразу после работы
    assert storage_left_open_pose(closed, {}) == []
    glance = _scene(15, lambda t: _safe(t, work=(2.0, 2.2)))          # рука мелькнула у сейфа
    assert storage_left_open_pose(glance, {}) == []
    no_zone = _scene(15, _safe)
    no_zone = SceneTimeline(**{**no_zone.__dict__, "zones": {**no_zone.zones, "safe": None}})
    assert storage_left_open_pose(no_zone, {}) == []


@pytest.mark.parametrize("name", sorted(POSE_DETECTORS))
def test_detectors_accept_common_params(name):
    scene = _scene(10, _h2h)
    params = {"p_on": 0.6, "p_off": 0.4, "smooth_points": 3, "merge_gap_sec": 5.0, "min_confidence": 0.5}
    assert isinstance(POSE_DETECTORS[name](scene, params), list)
