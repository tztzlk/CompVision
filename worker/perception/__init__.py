"""Восприятие: декодирование видео и покадровые детекции (без интерпретации правил).

Пакет импортируется без torch и OpenCV (его импортирует worker.events): тяжёлые зависимости
подгружаются лениво внутри функций. Подмодули:
  video     — кадры и JPEG через OpenCV (пути с кириллицей на Windows);
  geometry  — рамки в долях кадра; tracking — трекер людей по IoU и центрам;
  zones     — зоны стол / сейф / дверь / ноутбук по OWLv2; states — zero-shot-подписи и классы COCO;
  models    — RT-DETR, OWLv2, SigLIP (torch/transformers, лениво).
Сборка временного ряда состояний — worker.scene.
"""

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
    label: str              # класс объекта: person, cell phone, cup, ... (без идентификации лиц — ТЗ запрещает)
    score: float
    bbox_xyxy: tuple[float, float, float, float]   # в долях кадра 0..1
    track_id: int | None = None


@dataclass(frozen=True)
class FrameDetections:
    frame: Frame
    detections: list[Detection] = field(default_factory=list)


def extract_frames(video_path: Path, out_dir: Path, fps: float) -> list[Frame]:
    """Извлечь кадры с частотой fps в out_dir/000000.jpg … (OpenCV, таймкод по номеру кадра)."""
    from worker.perception.video import imwrite_unicode, iter_sampled_frames

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames: list[Frame] = []
    for k, ts, image in iter_sampled_frames(Path(video_path), fps):
        path = imwrite_unicode(out / f"{k:06d}.jpg", image)
        frames.append(Frame(index=k, timestamp_sec=round(ts, 3), path=path))
    return frames


def load_detector(models_dir: Path, device: str) -> object:
    """Загрузить детектор RT-DETR (COCO) из локальных весов в MODELS_DIR. Без сети."""
    from worker.perception.models import CocoDetector, model_path, resolve_device

    return CocoDetector(model_path(Path(models_dir), "person_detector"), resolve_device(device))


def detect(
    detector: object,
    frames: list[Frame],
    *,
    score_threshold: float = 0.3,
    batch_size: int = 8,
) -> list[FrameDetections]:
    """Прогнать детектор по кадрам с диска (кадр, который не прочитался, получает пустой список)."""
    from worker.perception.video import imread_unicode

    result: list[FrameDetections] = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        loaded = [(f, imread_unicode(f.path)) for f in chunk]
        readable = [(f, image) for f, image in loaded if image is not None]
        dets = detector([image for _, image in readable], score_threshold=score_threshold) if readable else []
        by_index = {f.index: d for (f, _), d in zip(readable, dets)}
        result.extend(FrameDetections(frame=f, detections=list(by_index.get(f.index, []))) for f in chunk)
    return result
