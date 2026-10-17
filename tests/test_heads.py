"""Головы поверх SigLIP (worker.perception.heads) и их влияние на сцену и R10 — без моделей."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from worker.events import detectors as det
from worker.perception.heads import PERSON_GEO_SCALE, crop_person, load_head, person_geo
from worker.perception.tracking import Track, TrackPoint
from worker.scene.builder import FrameRef, SceneConfig, _head_key, assemble_timeline
from worker.scene.contract import load_scene

FIXTURES = Path(__file__).parent / "fixtures"


def _save_head(path: Path, w, b: float) -> Path:
    np.savez(path, w=np.asarray(w, dtype=np.float32), b=np.float32(b), meta=json.dumps({"n": 1}))
    return path


def test_head_loads_and_predicts(tmp_path):
    _save_head(tmp_path / "h.npz", [2.0, -2.0], 0.0)
    head = load_head(tmp_path, "h.npz")
    p = head(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    assert p[0] > 0.85 and p[1] < 0.15 and abs(p[2] - 0.5) < 1e-6
    assert head.meta == {"n": 1}


def test_missing_head_is_none_not_error(tmp_path):
    assert load_head(tmp_path, "") is None
    assert load_head(tmp_path, "heads/нет.npz") is None


def test_person_features():
    g = person_geo((0.2, 0.1, 0.4, 0.5), 0.9, 0.4, (0.0, 0.45, 1.0, 0.8))
    assert np.allclose(g, [1.0, 0.05, 0.9, 0.5])
    assert len(g) == len(PERSON_GEO_SCALE)
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    assert crop_person(img, (0.2, 0.1, 0.4, 0.5)).shape == (49, 13, 3)
    assert crop_person(img, (0.2, 0.1, 0.21, 0.12)) is None          # меньше 8 px


# --- builder: голова правит только спорные точки --------------------------------------------------

N = 12
TIMES = [round(k / 3, 3) for k in range(N)]
DESK = (0.10, 0.45, 0.70, 0.70)
SEATED = (0.30, 0.30, 0.55, 0.75)
BEHIND_CASHIER = (0.33, 0.15, 0.45, 0.40)      # ноги закрыты кассиром: низ выше верха стола на 0,05
CORRIDOR = (0.60, 0.20, 0.66, 0.47)            # в коридоре за стеклом: низ ниже верха стола


def _scene(head_p: dict[int, float] | None):
    tracks = [Track(1, [TrackPoint(k, TIMES[k], SEATED, 0.9) for k in range(N)]),
              Track(2, [TrackPoint(k, TIMES[k], BEHIND_CASHIER, 0.9) for k in range(N)]),
              Track(3, [TrackPoint(k, TIMES[k], CORRIDOR, 0.6) for k in range(N)])]
    person_head = None
    if head_p is not None:
        person_head = [(p.frame_index, p.bbox, head_p[t.track_id]) for t in tracks for p in t.points[::3]]
    return assemble_timeline(
        video_id="Х-тест", duration_sec=TIMES[-1], frame_size=(464, 848),
        frames=[FrameRef(k, TIMES[k], f"frames/{k:06d}.jpg") for k in range(N)], tracks=tracks,
        objects=[[] for _ in range(N)], zones={"desk": DESK, "door": None, "safe": None, "laptop": None},
        state_samples={}, config=SceneConfig(fps=3.0), person_head=person_head)


def test_without_head_geometry_decides():
    # стоящий за кассиром «за стеклом» по низу рамки, человек в коридоре — «внутри»
    assert _scene(None).states[6].persons_in_room == 2


def test_confident_head_fixes_borderline_points():
    scene = _scene({1: 0.99, 2: 0.95, 3: 0.03})
    assert scene.states[6].persons_in_room == 2                      # кассир + стоящий за ним; коридор — нет
    others = {o.track_id for o in scene.states[6].others}
    assert 2 in others and 3 not in others


def test_unsure_head_keeps_geometry():
    assert _scene({1: 0.6, 2: 0.6, 3: 0.4}).states[6].persons_in_room == 2
    assert {o.track_id for o in _scene({1: 0.6, 2: 0.6, 3: 0.4}).states[6].others} == {3}


def test_head_key_matches_rounded_cache_boxes():
    box = (0.123456, 0.2, 0.3, 0.4)
    assert _head_key(5, box) == _head_key(5, [round(v, 4) for v in box])


# --- R10 с головой: head_* вместо параметров без приставки ----------------------------------------

def test_r10_with_head_uses_head_params(rules):
    """Голова видит два коротких законных открытия (Х-25: 8–11 и 15–18 с): склейка через 5 с без
    головы сделала бы одно на 10 с, head_merge_gap_sec 2 — нет. Открыт со 2-й секунды до конца: в фикстуре
    кассир у сейфа на 8–12 с, интервал 2–8 с (6 с) короче min_duration_sec и с головой не склеивается
    с 12–20 с — событие в 12 с (без головы склейка через 4 с дала бы 2 с)."""
    r10 = next(r for r in rules if r.id == "R10_storage_left_open")
    scene = load_scene(FIXTURES / "scene_X23.json")
    head_meta = {**scene.meta, "safe_head": {"name": "safe_open"}}

    def with_series(fn, meta):
        return dataclasses.replace(scene, meta=meta, states=tuple(
            dataclasses.replace(s, safe_open=fn(s.t)) for s in scene.states))

    two = lambda t: 0.97 if (8.0 <= t < 11.5 or 15.0 <= t < 18.5) else 0.02
    assert det.storage_left_open(with_series(two, head_meta), r10.params) == []
    late = lambda t: 0.97 if t >= 2.0 else 0.02
    events = det.storage_left_open(with_series(late, head_meta), r10.params)
    assert [round(e.timecode_sec) for e in events] == [12]
    no_head = det.storage_left_open(with_series(late, scene.meta), r10.params)
    assert [round(e.timecode_sec) for e in no_head] == [2]
