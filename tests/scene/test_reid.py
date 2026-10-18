"""Повторная идентификация кассира (SceneConfig.cashier_reid, worker.scene.reid) на синтетике.

Кассир (трек 1) сидит, идёт к двери, и трек обрывается у двери; на рабочем месте или в двери
появляется трек 4. Похожий эмбеддинг и разрыв не дольше reid_gap_sec — тот же кассир (выхода нет),
непохожий или позже — как без reid (выход кассира).
"""

from __future__ import annotations

import unittest

from tests.scene.test_builder import AT_DOOR, DESK, DOOR, ENTERING, FPS, N, SAFE, SEATED, TIMES, _track
from worker.scene.builder import FrameRef, SceneConfig, assemble_timeline
from worker.scene.contract import scene_to_dict
from worker.scene.reid import cosine, reid_stitch

CASHIER_EMB = [1.0, 0.0, 0.0]
SAME = [0.99, 0.1, 0.0]          # косинус 0,995
OTHER = [0.6, 0.8, 0.0]          # косинус 0,6
END = 14                         # последний кадр трека кассира (у двери), t = 4,667


def _tracks(start: int, box) -> list:
    return [
        _track(1, range(0, END + 1), lambda k: SEATED if k < 10 else AT_DOOR),
        _track(4, range(start, 30), lambda k: box),
    ]


def _reid(second) -> list:
    return [[1, CASHIER_EMB, CASHIER_EMB], [4, second, second]]


def _build(tracks, emb, on=True, **cfg):
    frames = [FrameRef(k, TIMES[k], f"frames/{k:06d}.jpg") for k in range(N)]
    return assemble_timeline(
        video_id="Х-тест", duration_sec=TIMES[-1], frame_size=(464, 848), frames=frames, tracks=tracks,
        objects=[[] for _ in frames], zones={"desk": DESK, "door": DOOR, "safe": SAFE, "laptop": None},
        state_samples={}, config=SceneConfig(fps=FPS, cashier_reid=on, **cfg), person_reid=emb)


def _left_room(scene) -> bool:
    return any(s.value("cashier_left_room") for s in scene.states)


class ReidTest(unittest.TestCase):
    def test_cosine(self) -> None:
        self.assertAlmostEqual(cosine(CASHIER_EMB, SAME), 0.995, places=3)
        self.assertEqual(cosine([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_similar_at_workplace_after_1s_is_stitched(self) -> None:
        tracks = _tracks(END + 3, SEATED)                    # через 1 с на рабочем месте
        base = _build(tracks, _reid(SAME), on=False)
        self.assertTrue(_left_room(base))                     # без reid — «вышла»
        scene = _build(tracks, _reid(SAME))
        self.assertFalse(_left_room(scene))
        self.assertEqual(scene.meta["scene"]["cashier_track_ids"], [1])
        self.assertEqual(scene.meta["scene"]["tracks"], 1)

    def test_similar_at_door_is_stitched(self) -> None:
        scene = _build(_tracks(END + 4, ENTERING), _reid(SAME))   # через 1,33 с в двери (stitch_tracks не берёт)
        self.assertFalse(_left_room(scene))
        self.assertFalse(any(s.value("cashier_returned") for s in scene.states))
        self.assertEqual(scene.meta["scene"]["tracks"], 1)

    def test_dissimilar_is_not_stitched(self) -> None:
        tracks = _tracks(END + 3, SEATED)
        scene = _build(tracks, _reid(OTHER))
        self.assertTrue(_left_room(scene))
        self.assertEqual(scene_to_dict(scene)["states"], scene_to_dict(_build(tracks, _reid(OTHER), on=False))["states"])

    def test_too_late_is_not_stitched(self) -> None:
        cfg = SceneConfig(fps=FPS)
        late = END + round((cfg.reid_gap_sec + 1.0) * FPS)   # через N + 1 с
        scene = _build(_tracks(late, SEATED), _reid(SAME))
        self.assertTrue(_left_room(scene))
        self.assertEqual(scene.meta["scene"]["tracks"], 2)

    def test_overlap_within_tolerance_is_stitched(self) -> None:
        """Х-35: новый трек на месте кассира начинается на кадр раньше, чем кончается старый."""
        tracks = [_track(1, range(0, 16), lambda k: SEATED), _track(4, range(15, 30), lambda k: SEATED)]
        self.assertEqual(_build(tracks, _reid(SAME), on=False).meta["scene"]["tracks"], 2)
        merged = _build(tracks, _reid(SAME))
        self.assertEqual(merged.meta["scene"]["tracks"], 1)
        self.assertEqual(merged.meta["scene"]["cashier_track_ids"], [1])

    def test_flag_off_is_unchanged(self) -> None:
        """Флаг выключен (по умолчанию): с эмбеддингами сцена та же, что без них."""
        self.assertFalse(SceneConfig().cashier_reid)
        tracks = _tracks(END + 3, SEATED)
        with_emb = _build(tracks, _reid(SAME), on=False)
        without = _build(tracks, None, on=False)
        self.assertEqual(scene_to_dict(with_emb), scene_to_dict(without))

    def test_reid_stitch_needs_cashier_and_embeddings(self) -> None:
        tracks = _tracks(END + 3, SEATED)
        cfg = SceneConfig(fps=FPS)
        self.assertIsNone(reid_stitch(tracks, tracks, [], None, None, _reid(SAME), cfg))     # кассира нет
        self.assertIsNone(reid_stitch(tracks, tracks, [1], SEATED, None, [[1, CASHIER_EMB, CASHIER_EMB]], cfg))
        out = reid_stitch(tracks, tracks, [1], SEATED, None, _reid(SAME), cfg)
        self.assertEqual([t.track_id for t in out], [1])
        self.assertEqual(len(tracks[0].points), END + 1)                                    # входные не меняются


if __name__ == "__main__":
    unittest.main()


class RebuildSetTest(unittest.TestCase):
    def test_parse_set(self) -> None:
        from tests.scene.test_scripts import _load_script

        mod = _load_script("rebuild_from_raw")
        self.assertEqual(mod.parse_overrides(["cashier_reid=true", "reid_min_cos=0.9", "reid_gap_sec=3"]),
                         {"cashier_reid": True, "reid_min_cos": 0.9, "reid_gap_sec": 3.0})
        with self.assertRaises(ValueError):
            mod.parse_overrides(["no_such_field=1"])
