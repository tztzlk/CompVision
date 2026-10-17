"""Чтение видео и JPEG через OpenCV (без ffmpeg-процессов).

Таймкод кадра = номер кадра / частота видео: ролики кейса ~30 к/с, почти постоянная частота.
На Windows cv2.imread/imwrite не понимают пути не в ASCII (файлы «Х-20.mov»), поэтому JPEG
пишутся и читаются через cv2.imencode/imdecode + numpy. Если VideoCapture не открыл файл
с кириллицей в пути, видео копируется во временный файл с ASCII-именем.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

__all__ = [
    "VideoError",
    "VideoInfo",
    "probe_video",
    "iter_sampled_frames",
    "read_frames_at",
    "imwrite_unicode",
    "imread_unicode",
]

_DEFAULT_FPS = 30.0


class VideoError(RuntimeError):
    """Видео не открывается или не читается."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    native_fps: float
    frame_count: int
    duration_sec: float


@contextmanager
def _capture(video_path: Path) -> Iterator["cv2.VideoCapture"]:
    path = Path(video_path)
    if not path.is_file():
        raise VideoError(f"видео не найдено: {path}")
    cap = cv2.VideoCapture(str(path))
    tmp_dir: Optional[Path] = None
    try:
        if not cap.isOpened() and not str(path).isascii():
            cap.release()
            tmp_dir = Path(tempfile.mkdtemp(prefix="compvision_video_"))
            tmp = tmp_dir / f"video{path.suffix.lower()}"
            shutil.copyfile(path, tmp)
            cap = cv2.VideoCapture(str(tmp))
        if not cap.isOpened():
            raise VideoError(f"OpenCV не открыл видео: {path}")
        yield cap
    finally:
        cap.release()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _fps(cap: "cv2.VideoCapture") -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if 1.0 <= fps <= 240.0 else _DEFAULT_FPS


def probe_video(video_path: Path) -> VideoInfo:
    """Размер, частота, число кадров и длительность (по метаданным контейнера)."""
    with _capture(video_path) as cap:
        fps = _fps(cap)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return VideoInfo(width=width, height=height, native_fps=fps, frame_count=max(0, count),
                     duration_sec=round(count / fps, 3) if count > 0 else 0.0)


def iter_sampled_frames(video_path: Path, fps: float) -> Iterator[tuple[int, float, np.ndarray]]:
    """Кадры с частотой fps: (порядковый номер k, таймкод в секундах, кадр BGR).

    Видео читается последовательно (grab без декодирования лишних кадров в BGR); берётся первый
    кадр, чей таймкод не раньше k / fps (с допуском в полкадра исходного видео).
    """
    if fps <= 0:
        raise ValueError("fps должно быть > 0")
    with _capture(video_path) as cap:
        native = _fps(cap)
        half_frame = 0.5 / native
        step = 1.0 / fps
        k, index, next_t = 0, -1, 0.0
        while cap.grab():
            index += 1
            ts = index / native
            if ts + half_frame < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            yield k, ts, frame
            k += 1
            next_t = k * step


def read_frames_at(video_path: Path, times: Sequence[float]) -> list[Optional[np.ndarray]]:
    """Кадры, ближайшие к заданным таймкодам (последовательное чтение — без неточного seek)."""
    if not times:
        return []
    order = sorted(range(len(times)), key=lambda i: times[i])
    result: list[Optional[np.ndarray]] = [None] * len(times)
    with _capture(video_path) as cap:
        native = _fps(cap)
        half_frame = 0.5 / native
        pos, index = 0, -1
        while pos < len(order) and cap.grab():
            index += 1
            ts = index / native
            if ts + half_frame < times[order[pos]]:
                continue
            ok, frame = cap.retrieve()
            while pos < len(order) and ts + half_frame >= times[order[pos]]:
                result[order[pos]] = frame if ok else None
                pos += 1
    return result


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 90) -> Path:
    """Записать JPEG по любому пути (включая кириллицу на Windows)."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise VideoError(f"не удалось закодировать JPEG: {path}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))
    return p


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Прочитать изображение по любому пути (None, если файла нет или он битый)."""
    p = Path(path)
    if not p.is_file():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
