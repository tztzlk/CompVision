"""Тесты контракта SceneTimeline (worker/scene/contract.py). Только stdlib.

Запуск из корня репозитория:
    PYTHONPATH=. python -m pytest tests/scene
    PYTHONPATH=. python -m unittest discover -s tests/scene
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worker.scene.contract import (
    PROB_FIELDS,
    ZONE_NAMES,
    Person,
    SceneContractError,
    SceneState,
    SceneTimeline,
    load_scene,
    save_scene,
    scene_from_dict,
    scene_to_dict,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs" / "scene_contract_example.json"


def _timeline(**overrides) -> SceneTimeline:
    states = (
        SceneState(t=0.0, frame="frames/000000.jpg", persons_in_room=1, cashier_present=1.0,
                   cashier_bbox=(0.3, 0.4, 0.6, 0.85), safe_open=0.0),
        SceneState(t=0.333, frame="frames/000001.jpg", persons_in_room=2, cashier_present=0.9,
                   cashier_bbox=(0.3, 0.4, 0.6, 0.85),
                   others=(Person(track_id=7, bbox=(0.0, 0.1, 0.2, 0.5), first_seen_t=0.333),),
                   person_entered=True, safe_open=None, cash_on_desk=0.75),
    )
    fields = dict(
        video_id="Х-20", duration_sec=11.6, fps_sampled=3.0, frame_size=(464, 848),
        zones={"desk": (0.05, 0.38, 0.62, 0.62), "door": None}, states=states,
        meta={"device": "cpu"},
    )
    fields.update(overrides)
    return SceneTimeline(**fields)


class ContractTest(unittest.TestCase):
    def test_example_file_matches_contract(self) -> None:
        scene = load_scene(EXAMPLE)
        self.assertEqual(scene.video_id, "Х-18")
        self.assertEqual(len(scene.states), 3)
        self.assertEqual(set(ZONE_NAMES), set(scene.zones))
        entered = [s.t for s in scene.states if s.person_entered]
        self.assertEqual(entered, [16.667])
        self.assertEqual(scene.states[2].persons_in_room, 3)
        self.assertEqual(scene.states[2].others[1].track_id, 7)
        self.assertIsNone(scene.states[0].laptop_unlocked)

    def test_roundtrip_keeps_cyrillic_and_nulls(self) -> None:
        scene = _timeline()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_scene(scene, Path(tmp) / "кэш" / "Х-20.json")
            raw = path.read_text(encoding="utf-8")
            self.assertIn('"video_id": "Х-20"', raw)          # ensure_ascii=False
            loaded = load_scene(path)
        self.assertEqual(loaded.video_id, "Х-20")
        self.assertEqual(loaded.frame_size, (464, 848))
        self.assertEqual(loaded.zones["desk"], (0.05, 0.38, 0.62, 0.62))
        self.assertIsNone(loaded.zones["door"])
        self.assertIsNone(loaded.zones["workplace"])           # недостающие зоны -> None
        first, second = loaded.states
        self.assertEqual(first.safe_open, 0.0)                 # 0 = «точно нет»
        self.assertIsNone(second.safe_open)                     # None = «не определено»
        self.assertTrue(second.person_entered)
        self.assertEqual(second.others[0].first_seen_t, 0.333)
        for name in PROB_FIELDS:
            self.assertIn(name, scene_to_dict(loaded)["states"][0])

    def test_unknown_fields_are_preserved(self) -> None:
        data = scene_to_dict(_timeline())
        data["states"][0]["hands_busy"] = 0.4
        data["camera_note"] = "угол 2"
        scene = scene_from_dict(data)
        self.assertEqual(scene.states[0].value("hands_busy"), 0.4)
        self.assertEqual(scene.states[0].value("cash_on_desk"), None)
        again = scene_to_dict(scene)
        self.assertEqual(again["states"][0]["hands_busy"], 0.4)
        self.assertEqual(again["camera_note"], "угол 2")

    def test_series_and_state_at(self) -> None:
        scene = _timeline()
        self.assertEqual(scene.series("persons_in_room"), [(0.0, 1), (0.333, 2)])
        self.assertEqual(scene.state_at(0.2).t, 0.333)
        self.assertEqual(scene.state_at(-5).t, 0.0)
        self.assertEqual(scene.state_at(99).t, 0.333)

    def test_validation_errors(self) -> None:
        base = scene_to_dict(_timeline())
        broken = {
            "вероятность > 1": lambda d: d["states"][0].update(cash_on_desk=1.5),
            "перевёрнутая рамка": lambda d: d["states"][0].update(cashier_bbox=[0.6, 0.4, 0.3, 0.85]),
            "t не по порядку": lambda d: d["states"].reverse(),
            "другая версия": lambda d: d.update(version=2),
            "нет states": lambda d: d.pop("states"),
            "строка вместо числа": lambda d: d["states"][0].update(persons_in_room="2"),
            "флаг не bool": lambda d: d["states"][0].update(person_entered=1),
            "t за концом видео": lambda d: d["states"][1].update(t=99.0),
        }
        for name, mutate in broken.items():
            with self.subTest(name):
                data = json.loads(json.dumps(base))
                mutate(data)
                with self.assertRaises(SceneContractError):
                    scene_from_dict(data)

    def test_save_refuses_invalid_scene(self) -> None:
        bad = _timeline(fps_sampled=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SceneContractError):
                save_scene(bad, Path(tmp) / "bad.json")
            self.assertFalse((Path(tmp) / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
