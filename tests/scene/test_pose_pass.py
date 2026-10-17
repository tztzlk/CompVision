"""Проход позы по сцене (worker/scene/pose_pass.py) и скрипты позы — с подменой модели, без GPU.

    PYTHONPATH=. python -m pytest tests/scene/test_pose_pass.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.scene import pose_synth as ps
from worker.perception.pose import Keypoints
from worker.scene.contract import Person, SceneState, SceneTimeline, load_scene, save_scene, scene_from_dict, scene_to_dict
from worker.scene.pose_features import FEATURE_NAMES, PoseConfig
from worker.scene.pose_pass import add_pose, pose_frames_from_scene, run_pose_pass

ROOT = Path(__file__).resolve().parents[2]
N = 12


def _scene() -> SceneTimeline:
    states = []
    for i in range(N):
        others = (Person(track_id=4, bbox=ps.box((170, 180, 300, 430)), first_seen_t=0.0),) if 3 <= i < 6 else ()
        states.append(SceneState(t=round(i / 3, 3), frame=f"frames/{i:06d}.jpg", persons_in_room=1 + len(others),
                                 cashier_present=0.95, cashier_bbox=ps.box(ps.CASHIER_BOX_PX), others=others,
                                 safe_open=0.1))
    return SceneTimeline(video_id="Х-поза", duration_sec=N / 3, fps_sampled=3.0, frame_size=ps.FRAME,
                         zones=ps.zones(), states=tuple(states))


class FakeEstimator:
    """Кассир — сидит, кисть на столе; другой человек тянет кисть к кисти кассира."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, images, boxes_per_image) -> list[list[Keypoints]]:
        self.calls.append(len(images))
        client = {i: (x, y - 150) for i, (x, y) in ps.SEATED.items()}
        client[9] = (250, 480)
        out = []
        for boxes in boxes_per_image:
            row = [ps.kp() if boxes[0] is not None else None]
            row += [ps.kp(base=client) for _ in boxes[1:]]
            out.append(row)
        return out


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AddPoseTest(unittest.TestCase):
    def test_fields_written_and_survive_save_load(self) -> None:
        est = FakeEstimator()
        scene = add_pose(_scene(), [object()] * N, est, PoseConfig())
        s0, s4 = scene.states[0], scene.states[4]
        self.assertEqual(len(s0.value("cashier_kp")), 17)
        self.assertEqual(s0.value("others_kp"), [])
        self.assertEqual([o["track_id"] for o in s4.value("others_kp")], [4])
        for name in FEATURE_NAMES:
            self.assertIn(name, s0.extra)
        self.assertEqual(s0.value("hand_to_hand"), 0.0)                 # никого рядом
        self.assertGreaterEqual(s4.value("hand_to_hand"), 0.6)          # кисть к кисти над столом
        self.assertEqual(s4.value("hand_to_hand_track"), 4)
        self.assertEqual(scene.meta["pose"]["states_with_cashier_kp"], N)
        self.assertEqual(sum(est.calls), N)

        with tempfile.TemporaryDirectory() as tmp:
            again = load_scene(save_scene(scene, Path(tmp) / "Х-поза.json"))
        self.assertEqual(again.states[4].value("hand_to_hand_track"), 4)
        self.assertEqual(again.states[4].value("cashier_kp"), s4.value("cashier_kp"))
        self.assertEqual(scene_to_dict(scene_from_dict(scene_to_dict(again))), scene_to_dict(again))

    def test_every_k_and_missing_frames(self) -> None:
        frames = [object() if i != 4 else None for i in range(N)]
        scene = add_pose(_scene(), frames, FakeEstimator(), PoseConfig(every_k=2))
        with_kp = [i for i, s in enumerate(scene.states) if s.value("cashier_kp") is not None]
        self.assertEqual(with_kp, [0, 2, 6, 8, 10])                    # кадр 4 не прочитан
        self.assertIsNone(scene.states[1].value("cashier_hand_at_face"))

    def test_recompute_from_saved_keypoints(self) -> None:
        scene = add_pose(_scene(), lambda i: object(), FakeEstimator())
        frames = pose_frames_from_scene(scene)
        self.assertEqual(frames[4].others[0].track_id, 4)
        self.assertIsNotNone(frames[4].others[0].kp)

    def test_contract_example_has_pose_fields(self) -> None:
        scene = load_scene(ROOT / "docs" / "scene_contract_example.json")
        last = scene.states[2]
        self.assertEqual(len(Keypoints.from_list(last.value("cashier_kp")).points), 17)
        self.assertEqual(last.value("hand_to_hand_track"), 3)
        frames = pose_frames_from_scene(scene)
        self.assertIsNone(frames[0].cashier_kp)                          # поза не считалась
        self.assertEqual([p.kp is not None for p in frames[2].others], [True, False])

    def test_run_pose_pass_without_weights_keeps_scene(self) -> None:
        scene = _scene()
        with tempfile.TemporaryDirectory() as tmp:
            out = run_pose_pass(scene, Path(tmp) / "frames", Path(tmp) / "models", "cpu")
        self.assertEqual(out.states, scene.states)
        self.assertIn("error", out.meta["pose"])


class PoseImportsTest(unittest.TestCase):
    def test_light_and_no_import_cycle(self) -> None:
        code = (
            "import importlib, sys\n"
            "import worker.events.pose_detectors as pd\n"           # раньше detectors: цикла нет
            "from worker.events.detectors import DETECTORS\n"
            "assert set(pd.POSE_DETECTORS) <= set(DETECTORS), sorted(DETECTORS)\n"
            "for name in ('worker.perception.pose', 'worker.scene.pose_features', 'worker.scene.pose_pass'):\n"
            "    importlib.import_module(name)\n"
            "print(sorted(m for m in ('torch', 'transformers', 'cv2', 'numpy') if m in sys.modules))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


class PoseScriptsTest(unittest.TestCase):
    def test_add_pose_to_cache_and_recompute_and_viz(self) -> None:
        from tests.scene.test_video import make_video

        add_script = _load_script("add_pose_to_cache")
        viz_script = _load_script("viz_pose")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset").mkdir()
            make_video(root / "dataset" / "Х-поза.mov", seconds=4.0, size=(116, 212))
            save_scene(_scene(), root / "cache" / "Х-поза.json")
            with mock.patch("worker.perception.pose.load_pose_model", return_value=FakeEstimator()):
                code = add_script.main(["--cache", str(root / "cache"), "--out", str(root / "pose"),
                                        "--dataset", str(root / "dataset"), "--device", "cpu"])
            self.assertEqual(code, 0)
            posed = load_scene(root / "pose" / "Х-поза.json")
            self.assertEqual(posed.states[4].value("hand_to_hand_track"), 4)

            self.assertEqual(add_script.main(["--out", str(root / "pose"), "--recompute"]), 0)
            self.assertEqual(load_scene(root / "pose" / "Х-поза.json").states[4].value("hand_to_hand_track"), 4)

            out = root / "viz" / "лист.jpg"
            code = viz_script.main(["--scene", str(root / "pose" / "Х-поза.json"),
                                    "--video", str(root / "dataset" / "Х-поза.mov"),
                                    "--every", "1", "--cols", "2", "--tile-width", "120", "--out", str(out)])
            self.assertEqual(code, 0)
            sheet = viz_script.imread_unicode(out)
        self.assertEqual(sheet.shape[1], 2 * 120)


if __name__ == "__main__":
    unittest.main()
