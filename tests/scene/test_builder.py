"""Тесты сборки SceneTimeline на синтетической сцене (только stdlib, без моделей).

Сцена 10 с при 3 к/с: кассир сидит у стола, на 5-й секунде через дверь входит посторонний
(его трек рвётся на одном кадре), за стеклом стоит человек, с 7-й секунды кассир уходит к двери.
"""

from __future__ import annotations

import unittest

from worker.perception import Detection
from worker.perception.tracking import Track, TrackPoint
from worker.scene.builder import FrameRef, SceneConfig, assemble_timeline, stitch_tracks
from worker.scene.contract import scene_from_dict, scene_to_dict

FPS = 3.0
N = 30
TIMES = [round(k / FPS, 3) for k in range(N)]
DESK = (0.10, 0.45, 0.70, 0.70)
DOOR = (0.00, 0.05, 0.25, 0.50)
SAFE = (0.70, 0.30, 0.85, 0.55)
SEATED = (0.30, 0.35, 0.55, 0.80)
AT_DOOR = (0.05, 0.20, 0.25, 0.62)
ENTERING = (0.02, 0.10, 0.20, 0.55)
BEHIND_GLASS = (0.30, 0.10, 0.36, 0.25)


def _track(track_id: int, frames: range, box_at) -> Track:
    return Track(track_id, [TrackPoint(k, TIMES[k], box_at(k), 0.9) for k in frames])


def _scene(*, tracks=None, objects=None, zones=None, samples=None, config=None):
    if tracks is None:
        tracks = [
            _track(1, range(0, N), lambda k: SEATED if k < 21 else AT_DOOR),          # кассир
            _track(2, range(15, 18), lambda k: ENTERING),                              # посторонний входит
            _track(5, range(19, N), lambda k: ENTERING),                               # ...и рвётся на кадре 18
            _track(3, range(0, N), lambda k: BEHIND_GLASS),                            # за стеклом
        ]
    frames = [FrameRef(k, TIMES[k], f"frames/{k:06d}.jpg") for k in range(N)]
    return assemble_timeline(
        video_id="Х-тест",
        duration_sec=TIMES[-1],
        frame_size=(464, 848),
        frames=frames,
        tracks=tracks,
        objects=objects if objects is not None else [[] for _ in range(N)],
        zones=zones if zones is not None else {"desk": DESK, "door": DOOR, "safe": SAFE, "laptop": None},
        state_samples=samples if samples is not None else {},
        config=config or SceneConfig(fps=FPS),
    )


class BuilderTest(unittest.TestCase):
    def test_roles_counts_and_entry(self) -> None:
        scene = _scene()
        self.assertEqual(scene.meta["scene"]["cashier_track_ids"], [1])
        self.assertIsNotNone(scene.zones["workplace"])
        self.assertEqual(scene.states[10].persons_in_room, 1)            # за стеклом не считается
        self.assertEqual(scene.states[16].persons_in_room, 2)
        self.assertEqual(scene.states[18].persons_in_room, 2)            # разрыв трека заполнен
        entered = [s.t for s in scene.states if s.person_entered]
        self.assertEqual(entered, [TIMES[15]])                           # один вход, без дубля после разрыва
        other = scene.states[20].others[0]
        self.assertEqual(other.track_id, 2)
        self.assertEqual(other.first_seen_t, TIMES[15])
        self.assertFalse(any(s.person_exited for s in scene.states))     # никто не ушёл до конца ролика

    def test_cashier_presence_drops_when_leaving(self) -> None:
        scene = _scene()
        self.assertEqual(scene.states[5].cashier_present, 1.0)
        self.assertEqual(scene.states[5].cashier_bbox, SEATED)
        self.assertLess(scene.states[27].cashier_present, 0.2)
        self.assertEqual(scene.states[27].cashier_bbox, AT_DOOR)

    def test_object_states_from_coco_and_zero_shot(self) -> None:
        objects = [[] for _ in range(N)]
        objects[4] = [Detection("cell phone", 0.8, (0.40, 0.50, 0.44, 0.56))]   # в руке кассира
        objects[8] = [Detection("handbag", 0.7, (0.20, 0.50, 0.30, 0.60))]      # на столе
        objects[9] = [Detection("cup", 0.6, (0.90, 0.90, 0.95, 0.95))]          # далеко от стола и рук
        samples = {"safe_open": [(0.0, 0.1), (3.0, 0.9)], "cash_on_desk": [(0.0, 0.2)]}
        scene = _scene(objects=objects, samples=samples)
        self.assertEqual(scene.states[4].phone_in_hand, 0.8)
        self.assertEqual(scene.states[5].phone_in_hand, 0.0)
        self.assertEqual(scene.states[8].bag_on_desk, 0.7)
        self.assertEqual(scene.states[9].cup_or_food, 0.0)
        self.assertEqual(scene.states[8].safe_open, 0.1)                 # до 3 с — последнее значение 0.1
        self.assertEqual(scene.states[9].safe_open, 0.9)
        self.assertEqual(scene.states[20].cash_on_desk, 0.2)
        self.assertIsNone(scene.states[0].laptop_unlocked)               # не оценивалось -> null
        self.assertIsNone(scene.states[0].outerwear_on_desk)

    def test_without_people_and_zones(self) -> None:
        scene = _scene(tracks=[], zones={})
        first = scene.states[0]
        self.assertEqual(first.persons_in_room, 0)
        self.assertIsNone(first.cashier_present)
        self.assertIsNone(first.phone_in_hand)
        self.assertIsNone(first.bag_on_desk)
        self.assertIsNone(scene.zones["workplace"])
        self.assertTrue(scene.meta["scene"]["warnings"])

    def test_result_passes_contract(self) -> None:
        scene = _scene()
        again = scene_from_dict(scene_to_dict(scene))
        self.assertEqual(len(again.states), N)
        self.assertEqual(again.video_id, "Х-тест")

    def test_stitch_does_not_mutate_input(self) -> None:
        a = _track(1, range(0, 3), lambda k: ENTERING)
        b = _track(2, range(4, 6), lambda k: ENTERING)
        merged = stitch_tracks([a, b], SceneConfig())
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].points), 5)
        self.assertEqual(len(a.points), 3)


if __name__ == "__main__":
    unittest.main()
