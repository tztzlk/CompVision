"""Тесты геометрии рамок и трекера людей (только stdlib)."""

from __future__ import annotations

import unittest

from worker.perception.geometry import center_dist, clip, expand, ioa, iou, median_box, nms, pad
from worker.perception.tracking import IoUTracker


class GeometryTest(unittest.TestCase):
    def test_iou_ioa(self) -> None:
        a = (0.0, 0.0, 0.5, 0.5)
        b = (0.25, 0.25, 0.75, 0.75)
        self.assertAlmostEqual(iou(a, b), 0.0625 / 0.4375)
        self.assertAlmostEqual(ioa(a, b), 0.25)
        self.assertEqual(iou(a, (0.6, 0.6, 0.9, 0.9)), 0.0)
        self.assertEqual(ioa((0.1, 0.1, 0.1, 0.2), b), 0.0)    # вырожденная рамка

    def assertBoxAlmostEqual(self, actual, expected) -> None:
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=9)

    def test_clip_expand_pad(self) -> None:
        self.assertEqual(clip((-0.1, 0.2, 1.3, 0.1)), (0.0, 0.1, 1.0, 0.2))
        self.assertBoxAlmostEqual(expand((0.2, 0.2, 0.4, 0.6), 0.5), (0.1, 0.0, 0.5, 0.8))
        self.assertBoxAlmostEqual(pad((0.5, 0.5, 0.6, 0.6), 0.1), (0.4, 0.4, 0.7, 0.7))

    def test_median_box_and_nms(self) -> None:
        boxes = [(0.1, 0.1, 0.3, 0.3), (0.12, 0.1, 0.32, 0.3), (0.9, 0.9, 1.0, 1.0)]
        self.assertEqual(median_box(boxes), (0.12, 0.1, 0.32, 0.3))
        self.assertIsNone(median_box([]))
        kept = nms([(boxes[0], 0.6), (boxes[1], 0.9), (boxes[2], 0.5)], 0.7)
        self.assertEqual(kept, [1, 2])


class TrackerTest(unittest.TestCase):
    def test_two_people_keep_ids(self) -> None:
        tracker = IoUTracker()
        ids = []
        for k in range(10):
            t = k / 3
            walker = (0.05 + 0.02 * k, 0.1, 0.25 + 0.02 * k, 0.55)   # идёт вправо
            sitter = (0.4, 0.35, 0.65, 0.8)                              # сидит
            ids.append(tracker.update(k, t, [(walker, 0.9), (sitter, 0.8)]))
        self.assertEqual(len(tracker.tracks), 2)
        self.assertTrue(all(row == ids[0] for row in ids))

    def test_lost_and_found_vs_new_track(self) -> None:
        tracker = IoUTracker(max_lost_sec=1.5)
        box = (0.4, 0.35, 0.65, 0.8)
        first = tracker.update(0, 0.0, [(box, 0.9)])[0]
        again = tracker.update(3, 1.0, [(box, 0.9)])[0]            # пропал на 1 с — тот же трек
        later = tracker.update(12, 4.0, [(box, 0.9)])[0]           # пропал на 3 с — новый трек
        self.assertEqual(first, again)
        self.assertNotEqual(first, later)

    def test_weak_unmatched_box_is_dropped(self) -> None:
        tracker = IoUTracker(min_new_score=0.5)
        self.assertEqual(tracker.update(0, 0.0, [((0.1, 0.1, 0.2, 0.3), 0.4)]), [None])
        self.assertEqual(tracker.tracks, [])

    def test_center_distance_matches_fast_walker(self) -> None:
        tracker = IoUTracker(iou_threshold=0.5, max_center_dist=0.12)
        a = (0.10, 0.1, 0.20, 0.5)
        b = (0.18, 0.1, 0.28, 0.5)                                   # IoU мал, центры рядом
        self.assertLess(iou(a, b), 0.5)
        self.assertLessEqual(center_dist(a, b), 0.12)
        first = tracker.update(0, 0.0, [(a, 0.9)])[0]
        second = tracker.update(1, 0.333, [(b, 0.9)])[0]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
