"""Прогон build_scene целиком на синтетическом видео с поддельными моделями (без torch и весов).

Проверяет связку pipeline -> builder -> контракт: кадры, трекинг, зоны, zero-shot, сохранение.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.scene.test_video import make_video
from worker.perception import Detection
from worker.scene import SceneConfig, build_scene, load_scene, save_scene

SEATED = (0.30, 0.35, 0.55, 0.80)
ZONES = {
    "a wooden office desk": (0.10, 0.45, 0.70, 0.70),
    "a metal safe": (0.70, 0.30, 0.85, 0.55),
    "a glass door": (0.00, 0.05, 0.25, 0.50),
    "a laptop": (0.45, 0.40, 0.60, 0.50),
}


class FakeBundle:
    device = "cpu"
    info = {"person_detector": "fake", "open_vocab": "fake", "zero_shot": "fake"}

    def __init__(self) -> None:
        self.frames_seen = 0
        self.crops_seen = 0

    def detector(self, images, *, score_threshold, labels):
        self.frames_seen += len(images)
        return [[
            Detection("person", 0.9, SEATED),
            Detection("cell phone", 0.7, (0.40, 0.50, 0.44, 0.56)),
            Detection("person", 0.2, (0.90, 0.90, 0.95, 0.99)),        # слабая рамка — отбрасывается
        ] for _ in images]

    def open_vocab(self, images, queries, *, score_threshold, top_k):
        return [[Detection(q, 0.5, ZONES[q]) for q in queries if q in ZONES] for _ in images]

    def zero_shot(self, crops, prompt_sets):
        self.crops_seen += len(crops)
        return [{name: (0.9 if name == "safe_open" else 0.1) for name in prompt_sets} for _ in crops]


class PipelineFakeTest(unittest.TestCase):
    def test_build_scene_end_to_end(self) -> None:
        bundle = FakeBundle()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "Х-тест.avi"
            make_video(video, seconds=4.0, size=(232, 424))
            work = root / "job"
            with mock.patch("worker.scene.pipeline.load_models", return_value=bundle):
                scene = build_scene(video, work, root / "models", "cpu", config=SceneConfig(fps=3.0))
            self.assertTrue((work / scene.states[0].frame).is_file())
            saved = save_scene(scene, root / "cache" / f"{scene.video_id}.json")
            loaded = load_scene(saved)

        self.assertEqual(loaded.video_id, "Х-тест")
        self.assertEqual(loaded.frame_size, (232, 424))
        self.assertEqual(len(loaded.states), 12)
        self.assertEqual(bundle.frames_seen, 12)
        self.assertEqual(loaded.zones["desk"], ZONES["a wooden office desk"])
        self.assertEqual(loaded.zones["door"], ZONES["a glass door"])
        self.assertIsNotNone(loaded.zones["workplace"])
        state = loaded.states[5]
        self.assertEqual(state.persons_in_room, 1)
        self.assertEqual(state.cashier_present, 1.0)
        self.assertEqual(state.phone_in_hand, 0.7)
        self.assertEqual(state.safe_open, 0.9)
        self.assertEqual(state.cash_on_desk, 0.1)
        self.assertEqual(state.laptop_unlocked, 0.1)
        self.assertIsNone(state.door_open)                     # выключено в SceneConfig по умолчанию
        self.assertFalse(any(s.person_entered for s in loaded.states))
        # zero-shot раз в секунду: 4 момента × 3 зоны (стол, сейф, ноутбук)
        self.assertEqual(bundle.crops_seen, 12)
        self.assertEqual(loaded.meta["device"], "cpu")
        self.assertIn("elapsed_sec", loaded.meta)
        self.assertIn("decode_detect_sec", loaded.meta["timings"])


if __name__ == "__main__":
    unittest.main()
