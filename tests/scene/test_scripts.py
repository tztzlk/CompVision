"""Тесты скриптов scripts/build_scene_cache.py и scripts/viz_scene.py без моделей."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.scene.test_builder import _scene
from tests.scene.test_video import make_video
from worker.scene import load_scene, save_scene

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildSceneCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.script = _load_script("build_scene_cache")

    def test_helpers(self) -> None:
        names = [Path(n) for n in ("Х-10.mov", "X-2.mov", "Х-3.mov", "X-1.mov")]
        self.assertEqual([p.stem for p in sorted(names, key=self.script.natural_key)], ["X-1", "X-2", "Х-3", "Х-10"])
        self.assertTrue(self.script.safe_dirname("Х-20").startswith("X-20_"))
        self.assertTrue(self.script.safe_dirname("Х-20").isascii())
        self.assertNotEqual(self.script.safe_dirname("Х-20"), self.script.safe_dirname("X-20"))

    def test_main_writes_cache_and_summary(self) -> None:
        def fake_build_scene(video, work, models_dir, device, *, video_id, config):
            if video_id == "Х-2":
                raise RuntimeError("битое видео")
            return _scene()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset" / "videos").mkdir(parents=True)
            for name in ("Х-1.mov", "Х-2.mov"):
                (root / "dataset" / "videos" / name).write_bytes(b"0")
            with mock.patch.object(self.script, "build_scene", side_effect=fake_build_scene):
                code = self.script.main([
                    "--dataset", str(root / "dataset"), "--out", str(root / "cache"),
                    "--work", str(root / "work"), "--device", "cpu",
                ])
            self.assertEqual(code, 1)                                    # одно видео с ошибкой
            cached = load_scene(root / "cache" / "Х-1.json")
            self.assertEqual(len(cached.states), 30)
            with (root / "cache" / "_summary.csv").open(encoding="utf-8-sig") as fh:
                rows = {r["video_id"]: r for r in csv.DictReader(fh)}
            self.assertEqual(rows["Х-1"]["status"], "ok")
            self.assertEqual(rows["Х-1"]["cashier_found"], "True")
            self.assertTrue(rows["Х-2"]["status"].startswith("error"))
            self.assertFalse((root / "work").exists() and any((root / "work").iterdir()))


class VizSceneTest(unittest.TestCase):
    def test_contact_sheet_from_video(self) -> None:
        script = _load_script("viz_scene")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "Х-тест.avi"
            make_video(video, seconds=10.0, size=(116, 212))
            scene_path = save_scene(_scene(), root / "Х-тест.json")
            out = root / "viz" / "лист.jpg"
            code = script.main(["--scene", str(scene_path), "--video", str(video), "--every", "2", "--cols", "3",
                                "--tile-width", "120", "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            sheet = script.imread_unicode(out)
        self.assertEqual(sheet.shape[1], 3 * 120)            # 3 столбца по 120 px
        self.assertEqual(sheet.shape[0], 2 * int(212 * 120 / 116))   # кадры на 0, 2, 4, 6, 8 с -> 2 ряда


if __name__ == "__main__":
    unittest.main()
