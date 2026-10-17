"""Геометрия позы (worker/perception/pose.py) и признаки (worker/scene/pose_features.py) на синтетике.

Без GPU, torch и моделей: точки задаются вручную (tests/scene/pose_synth.py).
    PYTHONPATH=. python -m pytest tests/scene/test_pose_features.py
"""

from __future__ import annotations

import math
import unittest

from tests.scene import pose_synth as ps
from worker.perception import pose
from worker.scene.pose_features import (
    FEATURE_NAMES,
    PersonPose,
    PoseConfig,
    PoseFrame,
    compute_features,
    frame_features,
    smooth3,
)

CFG = PoseConfig()


def _features(cashier, others=(), *, factor: float = 1.0, bbox=None, zones=None) -> dict:
    frame = PoseFrame(t=0.0, cashier_kp=cashier, cashier_bbox=bbox or ps.box(ps.CASHIER_BOX_PX, factor),
                      others=tuple(others))
    return frame_features(frame, zones if zones is not None else ps.zones(factor), ps.FRAME, CFG)


def _other(track_id: int, wrist_px, *, factor: float = 1.0, hidden=()) -> PersonPose:
    """Клиент напротив кассира (выше по кадру), одна кисть — в wrist_px."""
    base = {i: (x, y - 150) for i, (x, y) in ps.SEATED.items()}
    k = ps.kp({9: wrist_px, 10: (base[10][0] - 40, base[10][1] - 40)}, base=base, factor=factor, hidden=hidden)
    return PersonPose(track_id=track_id, bbox=ps.box((170, 180, 300, 430), factor), kp=k)


class GeometryTest(unittest.TestCase):
    def test_keypoint_constants_coco_order(self) -> None:
        self.assertEqual(pose.NUM_KEYPOINTS, 17)
        self.assertEqual(pose.KEYPOINT_NAMES[pose.LEFT_WRIST], "left_wrist")
        self.assertEqual((pose.LEFT_SHOULDER, pose.RIGHT_SHOULDER, pose.LEFT_HIP, pose.RIGHT_ANKLE), (5, 6, 11, 16))
        self.assertTrue(all(0 <= a < 17 and 0 <= b < 17 for a, b in pose.SKELETON))

    def test_keypoints_validation_and_roundtrip(self) -> None:
        k = ps.kp()
        again = pose.Keypoints.from_list(k.to_list())
        self.assertEqual(len(again.points), 17)
        self.assertAlmostEqual(again.points[9][0], k.points[9][0], places=3)
        self.assertIsNone(pose.Keypoints.from_list(None))
        with self.assertRaises(ValueError):
            pose.Keypoints(((0.1, 0.1, 0.9),) * 16)
        with self.assertRaises(ValueError):
            pose.Keypoints(((0.1, float("nan"), 0.9),) * 17)

    def test_distances_in_pixels_on_non_square_frame(self) -> None:
        # 0,1 кадра по x — 46,4 px, по y — 84,8 px
        self.assertAlmostEqual(pose.dist_px((0.0, 0.0), (0.1, 0.0), (464, 848)), 46.4)
        self.assertAlmostEqual(pose.dist_px((0.0, 0.0), (0.0, 0.1), (464, 848)), 84.8)

    def test_visible_wrists_head(self) -> None:
        k = ps.kp(hidden=(pose.LEFT_WRIST, pose.NOSE, pose.LEFT_EYE, pose.RIGHT_EYE))
        self.assertFalse(pose.visible(k, pose.LEFT_WRIST))
        self.assertTrue(pose.visible(k, pose.RIGHT_WRIST))
        self.assertEqual(len(pose.wrists(k)), 1)
        head = pose.head_point(k)                                   # только уши
        self.assertAlmostEqual(head[0] * 464, 232.0, places=3)
        self.assertIsNone(pose.head_point(ps.kp(hidden=pose.HEAD)))
        self.assertEqual(pose.wrists(None), [])

    def test_scale_fallbacks(self) -> None:
        self.assertAlmostEqual(pose.scale(ps.kp(), ps.FRAME), ps.S, places=3)          # ширина плеч
        torso = math.hypot(10, 80) * pose.TORSO_TO_SHOULDERS     # плечо–бедро одной стороны
        no_shoulder = ps.kp(hidden=(5,))                         # одно плечо не видно: торс по другой стороне
        self.assertAlmostEqual(pose.scale(no_shoulder, ps.FRAME), torso, places=3)
        sideways = ps.kp({5: (236, 400), 6: (228, 400)})          # плечи слились: торс
        self.assertAlmostEqual(pose.scale(sideways, ps.FRAME), math.hypot(16, 80) * pose.TORSO_TO_SHOULDERS, places=3)
        hidden_all = ps.kp(hidden=tuple(range(17)))
        bbox = ps.box(ps.CASHIER_BOX_PX)
        self.assertAlmostEqual(pose.scale(hidden_all, ps.FRAME, bbox), 250 * 0.25, places=3)
        self.assertIsNone(pose.scale(hidden_all, ps.FRAME))


class FeaturesTest(unittest.TestCase):
    def test_none_when_no_data(self) -> None:
        f = _features(None)
        self.assertTrue(all(f[n] is None for n in FEATURE_NAMES))
        f = _features(ps.kp(hidden=pose.WRISTS))                    # кисти не видны
        for name in ("cashier_hand_in_safe", "cashier_hands_on_desk", "hand_to_hand", "cashier_counting",
                     "cashier_hand_at_face"):
            self.assertIsNone(f[name], name)
        self.assertEqual(f["cashier_standing"], 0.0)                # ноги видны — сидит
        f = _features(ps.kp(), zones={"desk": None, "safe": None})  # зоны не найдены
        self.assertIsNone(f["cashier_hand_in_safe"])
        self.assertIsNone(f["cashier_hands_on_desk"])
        self.assertIsNone(f["hand_to_hand"])

    def test_wrist_in_safe_and_outside(self) -> None:
        self.assertEqual(_features(ps.kp())["cashier_hand_in_safe"], 0.0)
        inside = _features(ps.kp({10: (400, 350)}))
        self.assertEqual(inside["cashier_hand_in_safe"], 1.0)
        near = _features(ps.kp({10: (380 - 0.4 * ps.S, 350)}))     # левее зоны на 0,4·s — в расширении 0,5·s
        self.assertEqual(near["cashier_hand_in_safe"], 1.0)
        far = _features(ps.kp({10: (380 - 0.7 * ps.S, 350)}))
        self.assertEqual(far["cashier_hand_in_safe"], 0.0)

    def test_hands_on_desk_share(self) -> None:
        self.assertEqual(_features(ps.kp())["cashier_hands_on_desk"], 1.0)
        self.assertEqual(_features(ps.kp({10: (214, 380)}))["cashier_hands_on_desk"], 0.5)

    def test_hand_to_hand_threshold(self) -> None:
        cw = ps.SEATED[9]                                          # кисть кассира над столом (250, 490)

        def h2h(dist_s: float):
            return _features(ps.kp(), [_other(4, (cw[0], cw[1] - dist_s * ps.S))])

        close = h2h(0.25)
        self.assertGreaterEqual(close["hand_to_hand"], 0.9)
        self.assertEqual(close["hand_to_hand_track"], 4)
        self.assertAlmostEqual(h2h(0.5)["hand_to_hand"], 0.5, places=3)  # порог 0,5·s -> 0,5
        self.assertLess(h2h(0.7)["hand_to_hand"], 0.2)
        far = h2h(1.5)
        self.assertEqual(far["hand_to_hand"], 0.0)
        self.assertIsNone(far["hand_to_hand_track"])
        # середина между кистями не над столом
        high = _features(ps.kp({9: (250, 300)}), [_other(4, (250, 290))])
        self.assertEqual(high["hand_to_hand"], 0.0)
        # в комнате больше никого — точно нет; кисти другого не видны — нет данных
        self.assertEqual(_features(ps.kp())["hand_to_hand"], 0.0)
        self.assertIsNone(_features(ps.kp(), [_other(4, (250, 480), hidden=pose.WRISTS)])["hand_to_hand"])

    def test_hand_at_face(self) -> None:
        self.assertEqual(_features(ps.kp())["cashier_hand_at_face"], 0.0)
        at_face = _features(ps.kp({9: (240, 370)}))
        self.assertEqual(at_face["cashier_hand_at_face"], 1.0)
        self.assertIsNone(_features(ps.kp({9: (240, 370)}, hidden=pose.HEAD))["cashier_hand_at_face"])

    def test_standing_by_legs_and_by_bbox(self) -> None:
        self.assertEqual(_features(ps.kp())["cashier_standing"], 0.0)
        self.assertEqual(_features(ps.kp(ps.STANDING_LEGS))["cashier_standing"], 1.0)
        # ноги закрыты столом: рамка выше медианы своей высоты в 1,4 раза -> встал
        legs_hidden = ps.kp(hidden=(13, 14, 15, 16))
        seated_box = ps.box(ps.CASHIER_BOX_PX)
        tall_box = ps.box((180, 230, 290, 580))                    # 350 px против 250
        frames = [PoseFrame(t=i / 3, cashier_kp=legs_hidden, cashier_bbox=seated_box) for i in range(9)]
        frames += [PoseFrame(t=(9 + i) / 3, cashier_kp=legs_hidden, cashier_bbox=tall_box) for i in range(3)]
        feats = compute_features(frames, ps.zones(), ps.FRAME, CFG)
        self.assertEqual(feats[3]["cashier_standing"], 0.0)
        self.assertEqual(feats[10]["cashier_standing"], 1.0)

    def test_counting_window(self) -> None:
        def seq(moving: bool, apart: bool = False, others=()):
            frames = []
            for i in range(12):                                   # 4 с при 3 кадрах/с
                dx = (0.3 * ps.S if i % 2 else -0.3 * ps.S) if moving else 0.0
                right = (214 + dx, 490) if not apart else (160, 490)
                frames.append(PoseFrame(t=i / 3, cashier_kp=ps.kp({9: (250 + dx, 490), 10: right}),
                                        cashier_bbox=ps.box(ps.CASHIER_BOX_PX), others=tuple(others)))
            return compute_features(frames, ps.zones(), ps.FRAME, CFG)

        moving = seq(True)
        self.assertGreaterEqual(moving[6]["cashier_counting"], 0.6)
        self.assertEqual(seq(False)[6]["cashier_counting"], 0.0)   # кисти вместе, но неподвижны
        self.assertEqual(seq(True, apart=True)[6]["cashier_counting"], 0.0)  # кисти далеко друг от друга
        # кисть кассира в руке другого человека — это передача, а не пересчёт
        with_h2h = seq(True, others=[_other(4, (250, 485))])
        self.assertEqual(with_h2h[6]["cashier_counting"], 0.0)

    def test_scale_invariance(self) -> None:
        """Та же сцена в 2 раза крупнее (поза, рамки, зоны) даёт те же признаки."""
        def scene(factor: float):
            frames = []
            for i in range(12):
                dx = 0.3 * ps.S if i % 2 else -0.3 * ps.S
                overrides = {9: (250 + dx, 490), 10: (214 + dx, 490)}
                if i >= 8:
                    overrides = {**ps.STANDING_LEGS, 9: (240, 370), 10: (400, 350)}
                frames.append(PoseFrame(
                    t=i / 3, cashier_kp=ps.kp(overrides, factor=factor), cashier_bbox=ps.box(ps.CASHIER_BOX_PX, factor),
                    others=(_other(4, (250, 490 - 0.2 * ps.S), factor=factor),) if i in (3, 4) else ()))
            return compute_features(frames, ps.zones(factor), ps.FRAME, CFG)

        small, big = scene(0.5), scene(1.0)
        for i, (a, b) in enumerate(zip(small, big)):
            for name in FEATURE_NAMES:
                if a[name] is None or b[name] is None:
                    self.assertEqual(a[name], b[name], f"{name}@{i}")
                else:
                    self.assertTrue(math.isclose(a[name], b[name], abs_tol=0.02), f"{name}@{i}: {a[name]} != {b[name]}")
        self.assertTrue(any(r["cashier_counting"] for r in big))           # сцена что-то проверяет
        self.assertTrue(any(r["cashier_hand_in_safe"] for r in big))
        self.assertTrue(any(r["cashier_standing"] for r in big))
        self.assertTrue(any(r["hand_to_hand"] for r in big))

    def test_smooth3_median_keeps_none(self) -> None:
        self.assertEqual(smooth3([0.0, 1.0, 0.0, None, 1.0, 1.0]), [0.5, 0.0, 0.5, None, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
