"""Прогон build_scene целиком на синтетическом видео с поддельными моделями (без torch и весов).

Проверяет связку pipeline -> builder -> контракт: кадры, трекинг, зоны, zero-shot, сохранение.
"""

from __future__ import annotations

import contextlib
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

    def test_raw_cache_rebuilds_same_states(self) -> None:
        from worker.scene import scene_to_dict
        from worker.scene.raw_cache import load_raw, rebuild_scene

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "Х-сырой.avi"
            make_video(video, seconds=3.0, size=(232, 424))
            raw_path = root / "raw" / "Х-сырой.json"
            with mock.patch("worker.scene.pipeline.load_models", return_value=FakeBundle()):
                scene = build_scene(video, root / "job", root / "models", "cpu",
                                    config=SceneConfig(fps=3.0), raw_out=raw_path)
            rebuilt = rebuild_scene(load_raw(raw_path))
        self.assertEqual(scene_to_dict(rebuilt)["states"], scene_to_dict(scene)["states"])
        self.assertEqual(scene_to_dict(rebuilt)["zones"], scene_to_dict(scene)["zones"])
        self.assertTrue(rebuilt.meta["rebuilt_from_raw"])


class _ZeroShotWithEmbeddings:
    """zero_shot поддельной связки, у которого есть и image_embeddings (вход голов)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, crops, prompt_sets):
        return [{name: (0.9 if name == "safe_open" else 0.1) for name in prompt_sets} for _ in crops]

    def image_embeddings(self, crops):
        import numpy as np

        e = np.zeros((len(crops), self.dim), dtype=np.float32)
        e[:, 0] = 1.0
        return e


class PipelineHeadTest(unittest.TestCase):
    def _run(self, head_dim: int, emb_dim: int):
        import json

        import numpy as np

        bundle = FakeBundle()
        bundle.zero_shot = _ZeroShotWithEmbeddings(emb_dim)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models" / "heads").mkdir(parents=True)
            w = np.zeros(head_dim, dtype=np.float32)
            w[0] = -10.0                                   # эмбеддинг (1, 0, …) -> «закрыт»
            np.savez(root / "models" / "heads" / "safe_open.npz", w=w, b=np.float32(3.0), meta=json.dumps({}))
            video = root / "Х-голова.avi"
            make_video(video, seconds=3.0, size=(232, 424))
            with mock.patch("worker.scene.pipeline.load_models", return_value=bundle):
                return build_scene(video, root / "job", root / "models", "cpu",
                                   config=SceneConfig(fps=3.0, safe_head="heads/safe_open.npz"))

    def test_safe_head_replaces_zero_shot(self) -> None:
        scene = self._run(8, 8)
        self.assertLess(scene.states[4].safe_open, 0.01)          # sigmoid(-10 + 3)
        self.assertEqual(scene.meta["safe_head"]["name"], "safe_open")

    def test_wrong_head_size_falls_back_to_zero_shot(self) -> None:
        scene = self._run(5, 8)                                   # веса не той размерности
        self.assertEqual(scene.states[4].safe_open, 0.9)          # zero-shot, задача не упала
        self.assertIn("error", scene.meta["safe_head"])


class _FakeTower:
    """Дообученная башня: «сейф открыт» = 0,03, «наличные на столе» = 0,8 на любой вырезке."""
    name = "siglip_states"
    meta = {"trainable_blocks": 4}

    def __init__(self, tasks=("safe_open",)) -> None:
        self.tasks = {t: i for i, t in enumerate(tasks)}
        self.crops_seen = 0
        self.shapes = []

    def __call__(self, crops, tasks=None, *, batch_size=32):
        import numpy as np

        self.crops_seen += len(crops)
        self.shapes.extend(c.shape for c in crops)
        value = {"safe_open": 0.03, "desk_cash": 0.8}
        return {t: np.full(len(crops), value[t], dtype=np.float32) for t in tasks}


class PipelineFtTest(unittest.TestCase):
    """SceneConfig.safe_ft: по умолчанию выключен; без весов — сцена как без флага; с башней — её safe_open."""

    def _run(self, config: SceneConfig, tower=None, raw: bool = False):
        bundle = FakeBundle()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "Х-башня.avi"
            make_video(video, seconds=3.0, size=(232, 424))
            raw_path = root / "raw" / "Х-башня.json" if raw else None
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch("worker.scene.pipeline.load_models", return_value=bundle))
                if tower is not None:
                    stack.enter_context(mock.patch("worker.scene.pipeline.load_ft_tower", return_value=(tower, None)))
                scene = build_scene(video, root / "job", root / "models", "cpu", config=config, raw_out=raw_path)
            if raw:
                from worker.scene.raw_cache import load_raw

                return scene, load_raw(raw_path)
            return scene

    def test_on_by_default_and_can_be_disabled(self) -> None:
        scene = self._run(SceneConfig(fps=3.0))                    # весов в tmp нет — ошибка, не исключение
        self.assertIn("error", scene.meta["safe_ft"])
        self.assertIn("safe_ft_sec", scene.meta["timings"])
        off = self._run(SceneConfig(fps=3.0, safe_ft=""))
        self.assertNotIn("safe_ft", off.meta)
        self.assertNotIn("safe_ft_sec", off.meta["timings"])

    def test_old_raw_cache_rebuilt_with_flag_marks_error(self) -> None:
        """Сырой кэш без данных башни, пересобранный кодом с флагом (--code-config), — сцена та же,
        meta.safe_ft.error, R14 по OWLv2."""
        from worker.scene import scene_to_dict
        from worker.scene.raw_cache import rebuild_scene

        scene, raw = self._run(SceneConfig(fps=3.0, safe_ft=""), raw=True)
        rebuilt = rebuild_scene(raw, SceneConfig(fps=3.0))
        self.assertIn("error", rebuilt.meta["safe_ft"])
        self.assertEqual(scene_to_dict(rebuilt)["states"], scene_to_dict(scene)["states"])
        self.assertIsNone(rebuilt.states[4].value("desk_cash_ft"))

    def test_missing_weights_scene_as_without_flag(self) -> None:
        from worker.scene import scene_to_dict

        plain = self._run(SceneConfig(fps=3.0, safe_ft=""))
        flagged = self._run(SceneConfig(fps=3.0, safe_ft="ft/siglip_states"))
        self.assertIn("error", flagged.meta["safe_ft"])
        self.assertEqual(scene_to_dict(flagged)["states"], scene_to_dict(plain)["states"])

    def test_tower_replaces_safe_open(self) -> None:
        tower = _FakeTower()
        scene, raw = self._run(SceneConfig(fps=3.0, safe_ft="ft/siglip_states", safe_ft_fields=("safe_open",)),
                               tower=tower, raw=True)
        self.assertEqual(scene.states[4].safe_open, 0.03)
        self.assertEqual(scene.states[4].cash_on_desk, 0.1)        # общий SigLIP не подменён
        self.assertEqual(scene.meta["safe_ft"]["name"], "siglip_states")
        self.assertEqual(scene.meta["safe_ft"]["samples"], {"safe_open": 3})
        self.assertEqual(scene.meta["safe_ft"]["fields"], ["safe_open"])
        self.assertEqual(tower.crops_seen, 3)                      # вырезки сейфа раз в секунду
        samples = raw["state_samples"]
        self.assertEqual([p for _, p in samples["safe_open_zero_shot"]], [0.9, 0.9, 0.9])
        self.assertEqual([p for _, p in samples["safe_open"]], [0.03, 0.03, 0.03])
        self.assertNotIn("desk_cash", samples)
        self.assertEqual(raw["meta"]["safe_ft"]["name"], "siglip_states")

    def test_tower_desk_cash_separate_field(self) -> None:
        from worker.scene import scene_to_dict
        from worker.scene.raw_cache import rebuild_scene

        tower = _FakeTower(("safe_open", "desk_cash"))
        scene, raw = self._run(SceneConfig(fps=3.0, safe_ft="ft/siglip_states",
                                           safe_ft_fields=("safe_open", "desk_cash")), tower=tower, raw=True)
        self.assertEqual(scene.meta["safe_ft"]["fields"], ["safe_open", "desk_cash"])
        self.assertEqual(tower.crops_seen, 6)                      # сейф и стол раз в секунду
        self.assertEqual(scene.states[4].value("desk_cash_ft"), 0.8)
        plain = self._run(SceneConfig(fps=3.0))
        self.assertEqual(scene.states[4].value("desk_cash"), plain.states[4].value("desk_cash"))  # OWLv2 не тронут
        self.assertEqual([p for _, p in raw["state_samples"]["desk_cash"]], [0.8, 0.8, 0.8])
        rebuilt = rebuild_scene(raw)                               # сырой кэш воспроизводит поле
        self.assertEqual(scene_to_dict(rebuilt)["states"], scene_to_dict(scene)["states"])

    def test_default_fields_only_desk_cash(self) -> None:
        tower = _FakeTower(("safe_open", "desk_cash"))
        scene = self._run(SceneConfig(fps=3.0, safe_ft="ft/siglip_states"), tower=tower)
        self.assertEqual(scene.meta["safe_ft"]["fields"], ["desk_cash"])
        self.assertEqual(scene.states[4].safe_open, 0.9)            # сейф — как без башни
        self.assertEqual(scene.states[4].value("desk_cash_ft"), 0.8)
        self.assertEqual(tower.crops_seen, 3)                       # только стол

    def test_tower_without_requested_task_is_error(self) -> None:
        from worker.scene import scene_to_dict

        plain = self._run(SceneConfig(fps=3.0, safe_ft=""))
        scene = self._run(SceneConfig(fps=3.0, safe_ft="ft/siglip_states"), tower=_FakeTower(("safe_open",)))
        self.assertIn("error", scene.meta["safe_ft"])
        self.assertEqual(scene_to_dict(scene)["states"], scene_to_dict(plain)["states"])


if __name__ == "__main__":
    unittest.main()
