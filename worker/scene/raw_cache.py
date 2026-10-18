"""Сырой кэш восприятия: входы assemble_timeline одного видео в JSON (без кадров и без моделей).

Нужен, чтобы правки builder (склейка треков, вход через дверь, роль кассира) проверялись
без GPU: scripts/rebuild_from_raw.py пересобирает data/scene_cache из data/raw_cache за секунды.
Пишет worker.scene.pipeline.run_pipeline при raw_out (scripts/build_scene_cache.py --raw-out).
Если меняется выбор зон, выборки zero-shot устаревают: их пересчёт — только на GPU.

Модуль на stdlib: читается на машине без torch и OpenCV.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from worker.perception import Detection
from worker.perception.tracking import Track, TrackPoint
from worker.scene.builder import (FrameRef, SceneConfig, assemble_timeline, people_at, track_height,
                                  with_brightness, with_desk_items, with_ft_items, with_left_items)
from worker.scene.contract import SceneTimeline

__all__ = ["RAW_VERSION", "save_raw", "load_raw", "rebuild_scene"]

RAW_VERSION = 1


def _box(b: Sequence[float]) -> list[float]:
    return [round(float(v), 4) for v in b]


def _det(d: Detection) -> list[Any]:
    return [d.label, round(float(d.score), 4), _box(d.bbox_xyxy)]


def save_raw(
    path: Path,
    *,
    video_id: str,
    duration_sec: float,
    frame_size: tuple[int, int],
    frames: Sequence[FrameRef],
    tracks: Sequence[Track],
    objects: Sequence[Sequence[Detection]],
    zones: Mapping[str, Optional[Sequence[float]]],
    zone_info: Mapping[str, Any],
    owl_keyframes: Sequence[Sequence[Detection]],
    state_samples: Mapping[str, Sequence[tuple[float, Optional[float]]]],
    config: SceneConfig,
    meta: Optional[Mapping[str, Any]] = None,
    brightness: Optional[Sequence[float]] = None,
    item_checks: Optional[Sequence[tuple[float, Sequence[Detection]]]] = None,
    item_scan: Optional[Sequence[tuple[float, Sequence[Detection]]]] = None,
    person_head: Optional[Sequence[Sequence[Any]]] = None,
) -> Path:
    """Записать входы assemble_timeline атомарно (через временный файл)."""
    data = {
        "raw_version": RAW_VERSION,
        "video_id": video_id,
        "duration_sec": round(float(duration_sec), 3),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "frames": [[f.index, round(f.t, 3), f.frame] for f in frames],
        "tracks": [{"id": t.track_id, "points": [[p.frame_index, round(p.t, 3), _box(p.bbox), round(p.score, 4)]
                                                 for p in t.points]} for t in tracks],
        "objects": [[_det(d) for d in dets] for dets in objects],
        "zones": {k: (_box(v) if v is not None else None) for k, v in zones.items()},
        "zone_info": dict(zone_info),
        "owl_keyframes": [[_det(d) for d in dets] for dets in owl_keyframes],
        "state_samples": {k: [[round(t, 3), None if p is None else round(float(p), 4)] for t, p in v]
                          for k, v in state_samples.items()},
        "config": config.as_dict(),
        "meta": dict(meta or {}),
        "brightness": None if brightness is None else [round(float(b), 4) for b in brightness],
        "item_checks": [[round(float(t), 3), [_det(d) for d in dets]] for t, dets in (item_checks or [])],
        "item_scan": [[round(float(t), 3), [_det(d) for d in dets]] for t, dets in (item_scan or [])],
    }
    if person_head:
        data["person_head"] = [[int(fi), _box(b), round(float(pr), 4)] for fi, b, pr in person_head]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_raw(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("raw_version") != RAW_VERSION:
        raise ValueError(f"{path}: raw_version {data.get('raw_version')!r}, поддерживается {RAW_VERSION}")
    return data


def _config(raw: Mapping[str, Any], override: Optional[SceneConfig]) -> SceneConfig:
    if override is not None:
        return override
    known = {f.name for f in fields(SceneConfig)}
    return SceneConfig(**{k: v for k, v in raw.get("config", {}).items() if k in known})


def rebuild_scene(raw: Mapping[str, Any], config: Optional[SceneConfig] = None, *,
                  rezone: bool = False) -> SceneTimeline:
    """Собрать SceneTimeline текущим builder из сырого кэша (без моделей).

    rezone — зоны заново по сохранённым детекциям OWLv2 и COCO текущим worker.perception.zones.
    Выборки zero-shot при этом остаются от старых зон: состояния стола и сейфа на видео, где зона
    сменилась, верны только после пересборки на GPU.
    """
    def det(d: Sequence[Any]) -> Detection:
        return Detection(label=str(d[0]), score=float(d[1]), bbox_xyxy=tuple(d[2]))  # type: ignore[arg-type]

    frames = [FrameRef(index=int(i), t=float(t), frame=str(f)) for i, t, f in raw["frames"]]
    tracks = [Track(int(t["id"]), [TrackPoint(int(fi), float(tt), tuple(b), float(s))  # type: ignore[arg-type]
                                   for fi, tt, b, s in t["points"]]) for t in raw["tracks"]]
    objects = [[det(d) for d in dets] for dets in raw["objects"]]
    zones = {k: (tuple(v) if v is not None else None) for k, v in raw["zones"].items()}
    if rezone:
        from worker.perception.zones import infer_zones

        owl = [[det(d) for d in dets] for dets in raw.get("owl_keyframes", [])]
        people = [p.bbox for t in tracks for p in t.points]
        cfg = _config(raw, config)
        new_zones, _ = infer_zones(owl, objects, min_score=cfg.zone_min_score,
                                   person_boxes=people if cfg.desk_by_people else None,
                                   person_height=track_height(tracks))
        zones.update(new_zones)
    samples = {k: [(float(t), None if p is None else float(p)) for t, p in v] for k, v in raw["state_samples"].items()}
    meta = dict(raw.get("meta") or {})
    meta["rebuilt_from_raw"] = True
    timeline = assemble_timeline(
        video_id=raw["video_id"],
        duration_sec=raw["duration_sec"],
        frame_size=tuple(raw["frame_size"]),  # type: ignore[arg-type]
        frames=frames,
        tracks=tracks,
        objects=objects,
        zones=zones,  # type: ignore[arg-type]
        state_samples=samples,
        config=_config(raw, config),
        meta=meta,
        person_head=raw.get("person_head"),
    )
    brightness = raw.get("brightness")
    if brightness:
        timeline = with_brightness(timeline, brightness)
    checks = [(float(t), [det(d) for d in dets]) for t, dets in raw.get("item_checks") or []]
    timeline = with_left_items(timeline, checks)
    scan = [(float(t), [det(d) for d in dets]) for t, dets in raw.get("item_scan") or []]
    timeline = with_desk_items(timeline, scan, people_at(scan, frames, tracks))
    return with_ft_items(timeline, samples)
