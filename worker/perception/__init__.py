"""Восприятие: декодирование видео и покадровые детекции (без интерпретации правил)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Frame", "Detection", "FrameDetections", "extract_frames", "load_detector", "detect"]


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_sec: float
    path: Path              # JPEG в STORAGE_DIR/<job_id>/frames/


@dataclass(frozen=True)
class Detection:
    label: str              # класс объекта: person, hand, cash, ... (без идентификации лиц — ТЗ запрещает)
    score: float
    bbox_xyxy: tuple[float, float, float, float]   # в долях кадра 0..1
    track_id: int | None = None


@dataclass(frozen=True)
class FrameDetections:
    frame: Frame
    detections: list[Detection] = field(default_factory=list)


def extract_frames(video_path: Path, out_dir: Path, fps: float) -> list[Frame]:
    """ffmpeg: извлечь кадры с частотой fps в out_dir."""
    raise NotImplementedError


def load_detector(models_dir: Path, device: str) -> object:
    """Загрузить детектор из локальных весов в MODELS_DIR. Без сети."""
    raise NotImplementedError


def detect(detector: object, frames: list[Frame]) -> list[FrameDetections]:
    """Прогнать детектор по кадрам."""
    raise NotImplementedError
