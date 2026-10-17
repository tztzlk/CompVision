"""Головы жестов поверх точек позы (worker/scene/pose_heads.py): признаки окна, гибрид, загрузка .npz.

    PYTHONPATH=. python -m pytest tests/scene/test_pose_heads.py
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.scene import pose_synth as ps
from tests.scene.test_pose_pass import FakeEstimator, _scene
from worker.perception.pose import Keypoints
from worker.scene.pose_features import FEATURE_NAMES as GEOMETRY_NAMES
from worker.scene.pose_features import PersonPose, PoseConfig, PoseFrame, compute_features
from worker.scene.pose_heads import (
    FEATURE_NAMES, GEOMETRY_OF, GESTURES, GestureHead, apply_gesture_heads, hybrid_value, load_gesture_heads,
    state_features, window_features,
)
from worker.scene.pose_pass import add_pose, load_heads


def _mirror_box(b):
    return None if b is None else (1 - b[2], b[1], 1 - b[0], b[3])


def _mirror_kp(kp: Keypoints) -> Keypoints:
    return Keypoints(tuple((1 - x, y, s) for x, y, s in kp.points))


def _frames(factor: float = 1.0, mirror: bool = False) -> tuple[list[PoseFrame], dict]:
    """Кассир сидит, кисть тянется к сейфу и обратно; в середине — клиент с кистью у кисти кассира."""
    zones = ps.zones(factor)
    frames = []
    for i in range(9):
        reach = ps.wrist_at(2.4, -0.4) if 3 <= i <= 5 else ps.SEATED[9]
        kp = ps.kp({9: reach}, factor=factor)
        others = ()
        if i >= 6:
            client = {j: (x, y - 150) for j, (x, y) in ps.SEATED.items()}
            client[9] = (216, 492)
            others = (PersonPose(7, ps.box((170, 180, 300, 430), factor), ps.kp(base=client, factor=factor)),)
        frames.append(PoseFrame(t=round(i / 3, 3), cashier_kp=kp, cashier_bbox=ps.box(ps.CASHIER_BOX_PX, factor),
                                others=others))
    if mirror:
        zones = {k: _mirror_box(v) for k, v in zones.items()}
        frames = [PoseFrame(t=f.t, cashier_kp=_mirror_kp(f.cashier_kp), cashier_bbox=_mirror_box(f.cashier_bbox),
                            others=tuple(PersonPose(p.track_id, _mirror_box(p.bbox), _mirror_kp(p.kp)) for p in f.others))
                  for f in frames]
    return frames, zones


def _features(factor: float = 1.0, mirror: bool = False) -> list:
    frames, zones = _frames(factor, mirror)
    geometry = compute_features(frames, zones, ps.FRAME)
    return window_features(frames, geometry, zones, ps.FRAME)


def _write_heads(path: Path, *, mode: str = "band", prob: float = 0.9, names=FEATURE_NAMES, gestures=GESTURES) -> None:
    """Головы-заглушки: вероятность prob на любом входе (W = 0, b = logit(prob))."""
    n = len(names)
    data = {"feature_names": np.asarray(names)}
    heads = {}
    for g in gestures:
        data[f"{g}__mean"], data[f"{g}__std"] = np.zeros(n), np.ones(n)
        data[f"{g}__W0"], data[f"{g}__b0"] = np.zeros((n, 1)), np.array([math.log(prob / (1 - prob))])
        heads[g] = {"mode": mode, "layers": 1, "band_lo": 0.2, "band_hi": 0.8}
    data["meta"] = np.asarray(json.dumps({"version": 1, "gestures": list(gestures), "heads": heads}))
    np.savez(path, **data)


class WindowFeaturesTest(unittest.TestCase):
    def test_names_unique_and_vector_length(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        rows = _features()
        self.assertTrue(all(r is not None and len(r) == len(FEATURE_NAMES) for r in rows))
        self.assertFalse(any(math.isnan(v) for r in rows for v in r))

    def test_no_cashier_points_gives_none(self) -> None:
        frames, zones = _frames()
        frames[2] = PoseFrame(t=frames[2].t, cashier_kp=None, cashier_bbox=frames[2].cashier_bbox)
        self.assertIsNone(state_features(frames[2], zones, ps.FRAME))
        rows = window_features(frames, [{} for _ in frames], zones, ps.FRAME)
        self.assertIsNone(rows[2])
        self.assertIsNotNone(rows[3])

    def test_mirror_invariant(self) -> None:
        """Отражение кадра (дверь и сейф меняются сторонами) признаки не меняет: |dx| и min/max по рукам."""
        for a, b in zip(_features(), _features(mirror=True)):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_scale_invariant(self) -> None:
        """Та же сцена вдвое мельче — те же признаки: расстояния в единицах ширины плеч."""
        for a, b in zip(_features(), _features(factor=0.5)):
            np.testing.assert_allclose(a, b, atol=1e-3)

    def test_reach_to_safe_and_hand_to_client(self) -> None:
        rows = _features()
        safe = FEATURE_NAMES.index("arm_safe_min")
        other = FEATURE_NAMES.index("arm_other_min")
        self.assertLess(rows[4][safe], rows[0][safe])        # кисть у сейфа ближе к нему, чем сидя
        self.assertLess(rows[7][other], 1.0)                 # кисть клиента рядом
        self.assertEqual(rows[0][other], 6.0)                # клиента нет — «далеко»


class WindowRobustnessTest(unittest.TestCase):
    """Пропавшая в соседнем кадре точка — не жест: окно берёт только наблюдавшиеся значения."""

    def _still(self, hide_every_other: bool) -> list:
        zones = ps.zones()
        frames = []
        for i in range(9):
            hidden = (10,) if hide_every_other and i % 2 else ()
            frames.append(PoseFrame(t=round(i / 3, 3), cashier_kp=ps.kp(hidden=hidden),
                                    cashier_bbox=ps.box(ps.CASHIER_BOX_PX)))
        return window_features(frames, compute_features(frames, zones, ps.FRAME), zones, ps.FRAME)

    def test_flickering_wrist_is_not_motion(self) -> None:
        speed = FEATURE_NAMES.index("win_wrist_speed")
        self.assertTrue(all(r[speed] == 0.0 for r in self._still(True)))

    def test_missing_wrist_does_not_fill_window(self) -> None:
        zones = ps.zones()
        frames, _ = _frames()
        frames = frames[:3]
        frames[1] = PoseFrame(t=frames[1].t, cashier_kp=ps.kp(hidden=(9, 10)), cashier_bbox=frames[1].cashier_bbox)
        rows = window_features(frames, compute_features(frames, zones, ps.FRAME), zones, ps.FRAME)
        desk_max = FEATURE_NAMES.index("win_arm_desk_min_max")
        dy_min = FEATURE_NAMES.index("win_arm_dy_min_min")
        self.assertLess(rows[0][desk_max], 6.0)            # заглушка CAP соседа без кистей не попала в окно
        self.assertGreater(rows[0][dy_min], 0.0)           # и заглушка dy = 0 («кисть у плеч») тоже
        self.assertLess(rows[0][FEATURE_NAMES.index("win_vis_wrists_mean")], 1.0)


class HybridTest(unittest.TestCase):
    def _head(self, mode: str) -> GestureHead:
        return GestureHead("handover", None, None, (), mode, 0.2, 0.8)

    def test_modes(self) -> None:
        self.assertEqual(hybrid_value(0.5, 0.9, self._head("geometry")), 0.5)
        self.assertEqual(hybrid_value(0.5, 0.9, self._head("band")), 0.9)      # спорная геометрия
        self.assertEqual(hybrid_value(1.0, 0.1, self._head("band")), 1.0)      # уверенная геометрия остаётся
        self.assertIsNone(hybrid_value(None, 0.9, self._head("band")))
        self.assertEqual(hybrid_value(None, 0.9, self._head("fill")), 0.9)
        self.assertEqual(hybrid_value(1.0, 0.1, self._head("head")), 0.1)
        self.assertEqual(hybrid_value(0.5, None, self._head("head")), 0.5)     # голове нечем — геометрия


class LoadAndApplyTest(unittest.TestCase):
    def test_load_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.npz"
            _write_heads(path, prob=0.9)
            heads = load_gesture_heads(path)
        self.assertEqual(set(heads), set(GESTURES))
        p = heads["counting"].predict(np.zeros((3, len(FEATURE_NAMES))))
        np.testing.assert_allclose(p, 0.9, atol=1e-9)

    def test_geometry_mode_skipped_and_wrong_features_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.npz"
            _write_heads(path, mode="geometry")
            self.assertEqual(load_gesture_heads(path), {})
            _write_heads(path, names=FEATURE_NAMES[:-1])
            with self.assertRaises(ValueError):
                load_gesture_heads(path)

    def test_apply_band(self) -> None:
        frames, zones = _frames()
        geometry = compute_features(frames, zones, ps.FRAME)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.npz"
            _write_heads(path, mode="band", prob=0.9)
            heads = load_gesture_heads(path)
        rows = apply_gesture_heads(frames, geometry, zones, ps.FRAME, heads)
        for before, after in zip(geometry, rows):
            for g in GESTURES:
                field = GEOMETRY_OF[g]
                self.assertEqual(after[f"{field}_head"], 0.9)
                geo = before[field]
                expected = 0.9 if geo is not None and 0.2 < geo < 0.8 else geo
                self.assertEqual(after[field], expected)
            for name in GEOMETRY_NAMES:
                if name not in GEOMETRY_OF.values():
                    self.assertEqual(after[name], before[name])     # признаки без головы не трогаются


class PosePassWithHeadsTest(unittest.TestCase):
    def test_add_pose_with_and_without_heads(self) -> None:
        scene = _scene()
        frames = [object()] * len(scene.states)
        plain = add_pose(scene, frames, FakeEstimator())
        self.assertNotIn("gesture_heads", plain.meta["pose"])
        self.assertFalse(any(k.endswith("_head") for s in plain.states for k in s.extra))
        with tempfile.TemporaryDirectory() as tmp:
            _write_heads(Path(tmp) / "h.npz", mode="head", prob=0.7)
            heads, error = load_heads(PoseConfig(gesture_heads="h.npz"), Path(tmp))
        self.assertIsNone(error)
        with_heads = add_pose(scene, frames, FakeEstimator(), PoseConfig(), heads)
        self.assertEqual(with_heads.meta["pose"]["gesture_heads"], {g: "head" for g in sorted(GESTURES)})
        state = with_heads.states[0]
        self.assertEqual(state.value("hand_to_hand_head"), 0.7)
        self.assertEqual(state.value("hand_to_hand"), 0.7)              # режим head: значение головы

    def test_load_heads_errors_do_not_raise(self) -> None:
        self.assertEqual(load_heads(PoseConfig(), Path("/nonexistent")), ({}, None))
        heads, error = load_heads(PoseConfig(gesture_heads="missing.npz"), Path("/nonexistent"))
        self.assertEqual(heads, {})
        self.assertIn("heads:", error)


if __name__ == "__main__":
    unittest.main()
