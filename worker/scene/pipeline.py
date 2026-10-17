"""Оркестрация build_scene: видео -> кадры -> модели -> SceneTimeline.

Здесь живут тяжёлые зависимости (OpenCV, torch — через worker.perception.models); правила
восприятия сцены — в worker.scene.builder. Порядок:
  1. кадры с частотой SceneConfig.fps (JPEG в work_dir/frames/) + RT-DETR пачками;
  2. трекинг людей (IoU + центры);
  3. зоны: OWLv2 по нескольким ключевым кадрам (+ COCO как запасной источник);
  4. zero-shot SigLIP по вырезкам зон раз в state_every_sec;
  5. сборка SceneTimeline (builder).
Модели загружаются один раз на процесс (worker.perception.models.load_models).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from worker.perception import Detection, Frame
from worker.perception.geometry import BBox, nms
from worker.perception.models import load_models
from worker.perception.states import COCO_LABELS_OF_INTEREST, ZONE_CROP_MARGIN, crop_zone, enabled_states
from worker.perception.tracking import IoUTracker, Track
from worker.perception.video import VideoError, imread_unicode, imwrite_unicode, iter_sampled_frames, probe_video
from worker.perception.zones import all_zone_queries, infer_zones
from worker.scene.builder import FrameRef, SceneConfig, assemble_timeline
from worker.scene.contract import SceneTimeline

__all__ = ["FRAMES_SUBDIR", "run_pipeline", "pose_enabled"]

LOG = logging.getLogger("worker.scene")

FRAMES_SUBDIR = "frames"


def run_pipeline(
    video_path: Path,
    work_dir: Path,
    models_dir: Path,
    device: str,
    *,
    video_id: Optional[str] = None,
    config: Optional[SceneConfig] = None,
) -> SceneTimeline:
    """Видео -> SceneTimeline (см. worker.scene.build_scene)."""
    cfg = config or SceneConfig()
    started = time.monotonic()
    video_path, work_dir = Path(video_path), Path(work_dir)
    vid = video_id or video_path.stem
    info = probe_video(video_path)
    bundle = load_models(Path(models_dir), device)
    timings: dict[str, float] = {}

    mark = time.monotonic()
    frames, detections, size = _decode_and_detect(video_path, work_dir / FRAMES_SUBDIR, bundle, cfg)
    timings["decode_detect_sec"] = round(time.monotonic() - mark, 2)
    if not frames:
        raise VideoError(f"{video_path}: не удалось извлечь ни одного кадра")
    persons = [[d for d in dets if d.label == "person" and d.score >= cfg.person_score] for dets in detections]
    objects = [[d for d in dets if d.label != "person" and d.score >= cfg.object_score] for dets in detections]

    mark = time.monotonic()
    tracks = _track_people(frames, persons, cfg)
    timings["tracking_sec"] = round(time.monotonic() - mark, 2)

    mark = time.monotonic()
    zones, zone_info = _find_zones(frames, objects, bundle, cfg)
    timings["zones_sec"] = round(time.monotonic() - mark, 2)

    mark = time.monotonic()
    samples = _sample_states(frames, zones, bundle, cfg)
    timings["states_sec"] = round(time.monotonic() - mark, 2)

    refs = [FrameRef(index=pos, t=f.timestamp_sec, frame=f"{FRAMES_SUBDIR}/{f.path.name}") for pos, f in enumerate(frames)]
    frame_size = (info.width, info.height) if info.width > 0 and info.height > 0 else size
    meta: dict[str, Any] = {
        "device": bundle.device,
        "models": dict(bundle.info),
        "zones": zone_info,
        "native_fps": round(info.native_fps, 3),
        "frames_analyzed": len(frames),
        "timings": timings,
    }
    timeline = assemble_timeline(
        video_id=vid,
        duration_sec=info.duration_sec or frames[-1].timestamp_sec,
        frame_size=frame_size,
        frames=refs,
        tracks=tracks,
        objects=objects,
        zones=zones,
        state_samples=samples,
        config=cfg,
        meta=meta,
    )
    if pose_enabled():
        from worker.scene.pose_pass import run_pose_pass

        mark = time.monotonic()
        timeline = run_pose_pass(timeline, work_dir / FRAMES_SUBDIR, Path(models_dir), bundle.device)
        timings["pose_sec"] = round(time.monotonic() - mark, 2)
    elapsed = round(time.monotonic() - started, 2)
    LOG.info("сцена %s: %d кадров, %d треков, %.1f с (%s)", vid, len(frames), len(tracks), elapsed, bundle.device)
    return dataclasses.replace(timeline, meta={**timeline.meta, "timings": timings, "elapsed_sec": elapsed})


def pose_enabled() -> bool:
    """Проход позы (worker.scene.pose_pass) включается переменной окружения ENABLE_POSE=1."""
    return os.environ.get("ENABLE_POSE", "").strip() == "1"


def _decode_and_detect(
    video_path: Path, frames_dir: Path, bundle: Any, cfg: SceneConfig,
) -> tuple[list[Frame], list[list[Detection]], tuple[int, int]]:
    """Кадры с частотой cfg.fps -> JPEG + детекции RT-DETR (пачками по cfg.batch_size)."""
    frames: list[Frame] = []
    detections: list[list[Detection]] = []
    batch: list[Any] = []
    size = (0, 0)
    threshold = min(cfg.person_score, cfg.object_score)

    def flush() -> None:
        if batch:
            detections.extend(bundle.detector(batch, score_threshold=threshold, labels=COCO_LABELS_OF_INTEREST))
            batch.clear()

    for k, ts, image in iter_sampled_frames(video_path, cfg.fps):
        if not frames:
            size = (int(image.shape[1]), int(image.shape[0]))
        path = imwrite_unicode(frames_dir / f"{k:06d}.jpg", image, cfg.jpeg_quality)
        frames.append(Frame(index=k, timestamp_sec=round(ts, 3), path=path))
        batch.append(image)
        if len(batch) >= cfg.batch_size:
            flush()
    flush()
    if len(detections) != len(frames):
        raise RuntimeError(f"детектор вернул {len(detections)} списков на {len(frames)} кадров")
    return frames, detections, size


def _track_people(frames: Sequence[Frame], persons: Sequence[Sequence[Detection]], cfg: SceneConfig) -> list[Track]:
    tracker = IoUTracker(
        iou_threshold=cfg.track_iou,
        max_center_dist=cfg.track_center_dist,
        max_lost_sec=cfg.track_max_lost_sec,
        min_new_score=cfg.track_new_score,
    )
    for pos, (frame, dets) in enumerate(zip(frames, persons)):
        boxes = [(d.bbox_xyxy, d.score) for d in dets]
        keep = nms(boxes, cfg.person_nms_iou)
        tracker.update(pos, frame.timestamp_sec, [boxes[i] for i in keep])
    return tracker.tracks


def _keyframe_positions(count: int, keyframes: int) -> list[int]:
    """Равномерно разнесённые по видео позиции кадров (середины равных отрезков)."""
    if count <= 0 or keyframes <= 0:
        return []
    return sorted({min(count - 1, int((i + 0.5) * count / keyframes)) for i in range(keyframes)})


def _find_zones(
    frames: Sequence[Frame], objects: Sequence[Sequence[Detection]], bundle: Any, cfg: SceneConfig,
) -> tuple[dict[str, Optional[BBox]], dict[str, dict[str, object]]]:
    positions = _keyframe_positions(len(frames), cfg.zone_keyframes)
    images = [image for image in (imread_unicode(frames[p].path) for p in positions) if image is not None]
    owl = (bundle.open_vocab(images, all_zone_queries(), score_threshold=cfg.zone_min_score, top_k=cfg.zone_top_k)
           if images else [])
    return infer_zones(owl, objects, min_score=cfg.zone_min_score)


def _sample_states(
    frames: Sequence[Frame], zones: dict[str, Optional[BBox]], bundle: Any, cfg: SceneConfig,
) -> dict[str, list[tuple[float, Optional[float]]]]:
    """Zero-shot-оценки состояний по вырезкам зон раз в cfg.state_every_sec: {состояние: [(t, p)]}."""
    states = enabled_states(cfg)
    by_zone: dict[str, list[str]] = {}
    for name, spec in states.items():
        if zones.get(spec.zone) is not None:
            by_zone.setdefault(spec.zone, []).append(name)
    if not by_zone:
        return {}

    picks: list[int] = []
    next_t = 0.0
    for pos, frame in enumerate(frames):
        if frame.timestamp_sec + 1e-6 >= next_t:
            picks.append(pos)
            next_t = frame.timestamp_sec + cfg.state_every_sec

    samples: dict[str, list[tuple[float, Optional[float]]]] = {n: [] for names in by_zone.values() for n in names}
    for start in range(0, len(picks), cfg.state_batch_frames):
        chunk = picks[start:start + cfg.state_batch_frames]
        images = {pos: imread_unicode(frames[pos].path) for pos in chunk}
        for zone, names in by_zone.items():
            crops: list[Any] = []
            times: list[float] = []
            for pos in chunk:
                image = images[pos]
                crop = crop_zone(image, zones[zone], ZONE_CROP_MARGIN.get(zone, 0.1)) if image is not None else None
                if crop is not None:
                    crops.append(crop)
                    times.append(frames[pos].timestamp_sec)
            if not crops:
                continue
            probs = bundle.zero_shot(crops, {n: (states[n].positive, states[n].negative) for n in names})
            for t, p in zip(times, probs):
                for n in names:
                    samples[n].append((t, p.get(n)))
    return samples
