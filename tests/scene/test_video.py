"""Тесты чтения видео и JPEG (OpenCV) на синтетическом ролике с кириллицей в пути. Без моделей."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from worker.perception import extract_frames
from worker.perception.video import imread_unicode, imwrite_unicode, iter_sampled_frames, probe_video, read_frames_at

FPS = 30.0
SECONDS = 2.0
SIZE = (64, 96)          # ширина, высота
STEP = 4                 # яркость кадра = номер кадра * STEP


def make_video(path: Path, seconds: float = SECONDS, fps: float = FPS, size: tuple[int, int] = SIZE) -> int:
    """MJPG/AVI, яркость кадра растёт с номером кадра. Пишется под ASCII-именем и переименовывается:
    VideoWriter на Windows может не открыть путь с кириллицей."""
    tmp = path.with_name("tmp_ascii_video.avi")
    width, height = size
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV не умеет писать MJPG на этой машине")
    count = int(round(seconds * fps))
    for i in range(count):
        writer.write(np.full((height, width, 3), (i * STEP) % 256, dtype=np.uint8))
    writer.release()
    os.replace(tmp, path)
    return count


def frame_number(image: np.ndarray) -> int:
    return int(round(float(image.mean()) / STEP))


class VideoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "видео"
        self.dir.mkdir()
        self.video = self.dir / "Х-тест.avi"
        self.count = make_video(self.video)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_probe(self) -> None:
        info = probe_video(self.video)
        self.assertEqual((info.width, info.height), SIZE)
        self.assertAlmostEqual(info.native_fps, FPS, places=2)
        self.assertEqual(info.frame_count, self.count)
        self.assertAlmostEqual(info.duration_sec, SECONDS, places=2)

    def test_sampling_timestamps(self) -> None:
        sampled = list(iter_sampled_frames(self.video, 3.0))
        self.assertEqual([k for k, _, _ in sampled], list(range(6)))
        for k, ts, image in sampled:
            self.assertAlmostEqual(ts, k / 3.0, delta=0.5 / FPS)
            self.assertEqual(frame_number(image), round(ts * FPS))

    def test_extract_frames_to_cyrillic_dir(self) -> None:
        frames = extract_frames(self.video, self.dir / "кадры", 2.0)
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[1].path.name, "000001.jpg")
        self.assertAlmostEqual(frames[1].timestamp_sec, 0.5, delta=0.5 / FPS)
        image = imread_unicode(frames[1].path)
        self.assertIsNotNone(image)
        self.assertEqual(frame_number(image), 15)

    def test_read_frames_at_keeps_order(self) -> None:
        first, second = read_frames_at(self.video, [1.0, 0.0])
        self.assertEqual(frame_number(first), 30)
        self.assertEqual(frame_number(second), 0)

    def test_jpeg_roundtrip_unicode(self) -> None:
        image = np.full((20, 30, 3), 120, dtype=np.uint8)
        path = imwrite_unicode(self.dir / "кадр.jpg", image)
        loaded = imread_unicode(path)
        self.assertEqual(loaded.shape, image.shape)
        self.assertIsNone(imread_unicode(self.dir / "нет.jpg"))


if __name__ == "__main__":
    unittest.main()
