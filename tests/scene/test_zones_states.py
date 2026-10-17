"""Тесты выбора зон и zero-shot-состояний (чистая логика, без моделей)."""

from __future__ import annotations

import unittest

import numpy as np

from worker.perception import Detection
from worker.perception.states import COCO_LABELS_OF_INTEREST, STATE_PROMPTS, crop_zone, enabled_states
from worker.perception.zones import all_zone_queries, infer_zones
from worker.scene.builder import SceneConfig

WOODEN_DESK = (0.10, 0.45, 0.70, 0.70)
FRONT_TABLE = (0.00, 0.60, 0.60, 1.00)        # белый стол переднего плана, обрезан нижним краем


class ZonesTest(unittest.TestCase):
    def test_wooden_desk_beats_front_table(self) -> None:
        keyframe = [
            Detection("a wooden office desk", 0.30, WOODEN_DESK),
            Detection("a desk with a computer", 0.40, FRONT_TABLE),
            Detection("a metal safe", 0.25, (0.70, 0.30, 0.85, 0.55)),
        ]
        zones, info = infer_zones([keyframe] * 4, [[] for _ in range(10)])
        self.assertEqual(zones["desk"], WOODEN_DESK)
        self.assertEqual(info["desk"]["source"], "owlv2")
        self.assertEqual(zones["safe"], (0.70, 0.30, 0.85, 0.55))
        self.assertIsNone(zones["door"])
        self.assertIsNone(info["door"]["source"])

    def test_coco_fallback_uses_checkpoint_label_names(self) -> None:
        coco = [[Detection("diningtable", 0.8, WOODEN_DESK), Detection("laptop", 0.9, (0.45, 0.40, 0.60, 0.50))]] * 5
        coco += [[] for _ in range(5)]
        zones, info = infer_zones([[] for _ in range(4)], coco)
        self.assertEqual(zones["desk"], WOODEN_DESK)
        self.assertEqual(info["desk"]["source"], "coco")
        self.assertEqual(zones["laptop"], (0.45, 0.40, 0.60, 0.50))
        self.assertIn("diningtable", COCO_LABELS_OF_INTEREST)
        self.assertIn("cell phone", COCO_LABELS_OF_INTEREST)

    def test_rare_coco_hits_are_ignored(self) -> None:
        coco = [[Detection("laptop", 0.9, (0.45, 0.40, 0.60, 0.50))]] + [[] for _ in range(9)]
        zones, _ = infer_zones([], coco)
        self.assertIsNone(zones["laptop"])                   # 10% кадров < порога 20%

    def test_queries_unique(self) -> None:
        queries = all_zone_queries()
        self.assertEqual(len(queries), len(set(queries)))


class StatesTest(unittest.TestCase):
    def test_flags_disable_unreliable_states(self) -> None:
        default = enabled_states(SceneConfig())
        self.assertIn("safe_open", default)
        self.assertNotIn("door_open", default)
        self.assertNotIn("collection_bag", default)
        enabled = enabled_states(SceneConfig(enable_door_state=True))
        self.assertIn("door_open", enabled)
        for spec in STATE_PROMPTS.values():
            self.assertTrue(spec.positive and spec.negative)

    def test_crop_zone(self) -> None:
        image = np.zeros((848, 464, 3), dtype=np.uint8)
        crop = crop_zone(image, WOODEN_DESK, 0.0)
        self.assertEqual(crop.shape[:2], (int(np.ceil(0.70 * 848)) - int(0.45 * 848), int(np.ceil(0.70 * 464)) - int(0.10 * 464)))
        self.assertIsNone(crop_zone(image, (0.5, 0.5, 0.505, 0.505), 0.0))


if __name__ == "__main__":
    unittest.main()


class PromptsV2Test(unittest.TestCase):
    def test_safe_prompts_are_minimal_pairs(self) -> None:
        from worker.perception.states import STATE_PROMPTS_V2

        spec = enabled_states(SceneConfig(state_prompts="v2"))["safe_open"]
        self.assertIs(spec, STATE_PROMPTS_V2["safe_open"])
        for pos, neg in zip(spec.positive, spec.negative):
            self.assertEqual(pos.replace("open", "closed"), neg)
        self.assertEqual(set(enabled_states(SceneConfig(state_prompts="v2"))), set(enabled_states(SceneConfig())))

    def test_desk_negatives_have_no_person(self) -> None:
        for name in ("cash_on_desk", "bag_on_desk", "outerwear_on_desk"):
            spec = enabled_states(SceneConfig(state_prompts="v2"))[name]
            self.assertFalse(any("person" in p for p in (*spec.positive, *spec.negative)))

    def test_mask_people(self) -> None:
        import numpy as np

        from worker.perception import Detection
        from worker.scene.pipeline import _mask_people

        image = np.zeros((100, 50, 3), dtype=np.uint8)
        out = _mask_people(image, [Detection("person", 0.9, (0.2, 0.5, 0.6, 1.0))])
        self.assertEqual(int(out[60, 20, 0]), 128)
        self.assertEqual(int(out[10, 20, 0]), 0)
        self.assertEqual(int(image[60, 20, 0]), 0)            # исходный кадр не тронут


class DeskByPeopleTest(unittest.TestCase):
    def test_working_desk_beats_bigger_empty_desk(self) -> None:
        """Х-20: OWLv2 выше оценивает большой пустой белый стол на переднем плане; люди сидят у деревянного."""
        from worker.perception.zones import desk_candidates, pick_desk_by_people

        white = (0.0, 0.62, 0.75, 0.90)
        wooden = (0.08, 0.36, 0.62, 0.55)
        owl = [[Detection("a wooden office desk", 0.5, white), Detection("a desk with a computer", 0.3, wooden)]] * 4
        cands = desk_candidates(owl)
        self.assertEqual(cands[0][0], white)
        cashier = [(0.25, 0.25, 0.45, 0.50)] * 30
        self.assertEqual(pick_desk_by_people(cands, cashier), wooden)
        self.assertIsNone(pick_desk_by_people(cands[:1], cashier))


class ZoneScoringTest(unittest.TestCase):
    def test_desk_behind_glass_does_not_win_by_two_queries(self) -> None:
        """Х-20: стол соседнего офиса совпал с обоими запросами (0,40 + 0,35), рабочий — с одним (0,54)."""
        working = (0.14, 0.41, 0.71, 0.63)
        behind_glass = (0.29, 0.08, 0.46, 0.11)
        owl = [[Detection("a wooden office desk", 0.54, working),
                Detection("a wooden office desk", 0.40, behind_glass),
                Detection("a desk with a computer", 0.35, behind_glass)] for _ in range(4)]
        zones, _ = infer_zones(owl, [[] for _ in range(10)])
        self.assertEqual(zones["desk"], working)

    def test_safe_is_best_single_detection(self) -> None:
        """Х-3: белую тумбу «a metal safe» находит слабо, но на всех кадрах; серый шкаф — уверенно на одном."""
        drawer = (0.0, 0.56, 0.17, 0.70)
        cabinet = (0.66, 0.42, 0.89, 0.59)
        owl = [[Detection("a metal safe", 0.14, drawer)] for _ in range(4)]
        owl[1] = owl[1] + [Detection("a grey metal cabinet", 0.31, cabinet)]
        zones, _ = infer_zones(owl, [[] for _ in range(10)])
        self.assertEqual(zones["safe"], cabinet)


class SafeGeometryTest(unittest.TestCase):
    """Сейф по геометрии (замер 18.10, data/ml_results/zones_wrong.jpg)."""

    DESK = (0.08, 0.43, 0.65, 0.66)

    def _zones(self, dets):
        from worker.perception.zones import infer_zones

        desk = Detection("a wooden office desk", 0.6, self.DESK)
        return infer_zones([[desk, *dets] for _ in range(4)], [[] for _ in range(10)], person_height=0.24)

    def test_machine_on_desk_is_not_safe(self) -> None:
        """Х-31: чёрная машинка на столешнице похожа на «металлический шкаф» сильнее настоящего сейфа."""
        machine = (0.30, 0.47, 0.42, 0.58)
        cabinet = (0.46, 0.33, 0.61, 0.44)
        zones, _ = self._zones([Detection("a grey metal cabinet", 0.40, machine),
                                Detection("a grey metal cabinet", 0.32, cabinet)])
        self.assertEqual(zones["safe"], cabinet)

    def test_printer_like_candidate_is_not_safe(self) -> None:
        printer = (0.46, 0.38, 0.58, 0.45)
        zones, info = self._zones([Detection("a metal safe", 0.20, printer), Detection("a printer", 0.53, printer)])
        self.assertIsNone(zones["safe"])
        self.assertIn("decoy", info["safe"]["reason"])

    def test_stand_under_cabinet_and_cabinet_behind_glass(self) -> None:
        """Х-37: серый шкаф на белой тумбе; шкаф в соседнем офисе за стеклом (низ выше края стола)."""
        stand = (0.67, 0.52, 0.95, 0.67)
        cabinet = (0.71, 0.41, 0.97, 0.57)
        behind = (0.19, 0.18, 0.30, 0.32)
        zones, _ = self._zones([Detection("a grey metal cabinet", 0.33, stand),
                                Detection("a grey metal cabinet", 0.20, cabinet),
                                Detection("a grey metal cabinet", 0.31, behind)])
        self.assertEqual(zones["safe"], cabinet)

