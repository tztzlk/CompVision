"""Оркестрация build_scene: видео -> кадры -> модели -> SceneTimeline.

Здесь живут тяжёлые зависимости (OpenCV, torch — через worker.perception.models); правила
восприятия сцены — в worker.scene.builder. Порядок:
  1. кадры с частотой SceneConfig.fps (JPEG в work_dir/frames/) + RT-DETR пачками;
  2. трекинг людей (IoU + центры);
  3. зоны: OWLv2 по нескольким ключевым кадрам (+ COCO как запасной источник);
  4. zero-shot SigLIP по вырезкам зон раз в state_every_sec; вид двери против начала ролика (R15, без моделей);
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
from worker.perception.door import door_band, door_diffs, door_patch
from worker.perception.ft_tower import ft_tower_info, load_ft_tower
from worker.perception.geometry import BBox, nms
from worker.perception.heads import PERSON_GEO_SCALE, LinearHead, crop_person, load_head, person_geo
from worker.perception.models import load_models
from worker.perception.states import COCO_LABELS_OF_INTEREST, ZONE_CROP_MARGIN, crop_zone, enabled_states
from worker.perception.tracking import IoUTracker, Track
from worker.perception.video import VideoError, imread_unicode, imwrite_unicode, iter_sampled_frames, probe_video
from worker.perception.zones import all_zone_queries, infer_zones
from worker.scene.builder import (FrameRef, SceneConfig, ITEM_QUERIES, SCAN_QUERIES, assemble_timeline, item_check_times, track_height,
                                  with_desk_items, with_ft_items, people_at,
                                  with_brightness, with_door_change, with_left_items)
from worker.scene.contract import SceneTimeline

__all__ = ["FRAMES_SUBDIR", "run_pipeline", "pose_enabled", "apply_safe_ft", "FT_FIELDS", "door_signal"]

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
    raw_out: Optional[Path] = None,
) -> SceneTimeline:
    """Видео -> SceneTimeline (см. worker.scene.build_scene).

    raw_out — файл сырого кэша (worker.scene.raw_cache): входы сборки сцены для правок builder без GPU.
    """
    cfg = config or SceneConfig()
    started = time.monotonic()
    video_path, work_dir = Path(video_path), Path(work_dir)
    vid = video_id or video_path.stem
    info = probe_video(video_path)
    bundle = load_models(Path(models_dir), device)
    timings: dict[str, float] = {}

    mark = time.monotonic()
    frames, detections, size, brightness = _decode_and_detect(video_path, work_dir / FRAMES_SUBDIR, bundle, cfg)
    timings["decode_detect_sec"] = round(time.monotonic() - mark, 2)
    if not frames:
        raise VideoError(f"{video_path}: не удалось извлечь ни одного кадра")
    persons = [[d for d in dets if d.label == "person" and d.score >= cfg.person_score] for dets in detections]
    objects = [[d for d in dets if d.label != "person" and d.score >= cfg.object_score] for dets in detections]

    mark = time.monotonic()
    tracks = _track_people(frames, persons, cfg)
    timings["tracking_sec"] = round(time.monotonic() - mark, 2)

    mark = time.monotonic()
    zones, zone_info, owl = _find_zones(frames, objects, tracks, bundle, cfg)
    timings["zones_sec"] = round(time.monotonic() - mark, 2)

    door_diff: Optional[list[Optional[float]]] = None
    if cfg.door_change:
        mark = time.monotonic()
        door_diff = door_signal(frames, persons, zones.get("door"), cfg)
        timings["door_sec"] = round(time.monotonic() - mark, 2)

    mark = time.monotonic()
    safe_head = load_head(Path(models_dir), cfg.safe_head)
    head_errors: list[str] = []
    samples = _sample_states(frames, zones, bundle, cfg, persons, safe_head=safe_head, head_errors=head_errors)
    if head_errors:
        safe_head = None
    timings["states_sec"] = round(time.monotonic() - mark, 2)

    ft_meta: Optional[dict[str, Any]] = None
    if cfg.safe_ft:
        mark = time.monotonic()
        ft_meta = apply_safe_ft(samples, frames, zones, Path(models_dir), bundle.device, cfg)
        timings["safe_ft_sec"] = round(time.monotonic() - mark, 2)

    person_head = None
    inside_head = load_head(Path(models_dir), cfg.inside_head)
    if inside_head is not None:
        mark = time.monotonic()
        try:
            person_head = person_inside_probs(frames, tracks, zones, bundle, inside_head)
        except Exception as exc:     # не та размерность весов и т.п.: внутри/за стеклом — по геометрии
            LOG.warning("голова %s: %s — внутри/за стеклом по геометрии", inside_head.name, exc)
            head_errors.append(str(exc))
            inside_head = None
        timings["inside_head_sec"] = round(time.monotonic() - mark, 2)

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
    if cfg.inside_head:
        meta["inside_head"] = ({"name": inside_head.name, "points": len(person_head or [])}
                               if inside_head is not None else {"error": "; ".join(head_errors) or f"не загружена: {cfg.inside_head}"})
    if cfg.safe_head:
        meta["safe_head"] = ({"name": safe_head.name, **{k: safe_head.meta.get(k) for k in ("n", "pos", "cv")}}
                             if safe_head is not None else {"error": "; ".join(head_errors) or f"не загружена: {cfg.safe_head}"})
    if ft_meta is not None:
        meta["safe_ft"] = ft_meta
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
        person_head=person_head,
    )
    timeline = with_brightness(timeline, brightness)
    if door_diff is not None:
        timeline = with_door_change(timeline, door_diff, cfg)
    mark = time.monotonic()
    item_checks = check_left_items(timeline, frames, bundle, cfg)
    timeline = with_left_items(timeline, item_checks)
    item_scan = scan_desk_items(frames, bundle, cfg)
    timeline = with_desk_items(timeline, item_scan, people_at(item_scan, frames, tracks))
    timeline = with_ft_items(timeline, samples)
    timings["items_sec"] = round(time.monotonic() - mark, 2)
    if raw_out is not None:
        from worker.scene.raw_cache import save_raw

        save_raw(raw_out, video_id=vid, duration_sec=info.duration_sec or frames[-1].timestamp_sec,
                 frame_size=frame_size, frames=refs, tracks=tracks, objects=objects, zones=zones,
                 zone_info=zone_info, owl_keyframes=owl, state_samples=samples, config=cfg, meta=meta,
                 brightness=brightness, item_checks=item_checks, item_scan=item_scan, person_head=person_head,
                 door_diff=door_diff)
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
) -> tuple[list[Frame], list[list[Detection]], tuple[int, int], list[float]]:
    """Кадры с частотой cfg.fps -> JPEG + детекции RT-DETR (пачками по cfg.batch_size) + средняя яркость кадра 0..1."""
    frames: list[Frame] = []
    brightness: list[float] = []
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
        brightness.append(float(image.mean()) / 255.0)
        batch.append(image)
        if len(batch) >= cfg.batch_size:
            flush()
    flush()
    if len(detections) != len(frames):
        raise RuntimeError(f"детектор вернул {len(detections)} списков на {len(frames)} кадров")
    return frames, detections, size, brightness


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
    frames: Sequence[Frame], objects: Sequence[Sequence[Detection]], tracks: Sequence[Track], bundle: Any,
    cfg: SceneConfig,
) -> tuple[dict[str, Optional[BBox]], dict[str, dict[str, object]], list[list[Detection]]]:
    """Зоны, сведения об их источнике и сырые детекции OWLv2 по ключевым кадрам (для сырого кэша)."""
    positions = _keyframe_positions(len(frames), cfg.zone_keyframes)
    images = [image for image in (imread_unicode(frames[p].path) for p in positions) if image is not None]
    owl = (bundle.open_vocab(images, all_zone_queries(), score_threshold=cfg.zone_owl_score, top_k=cfg.zone_top_k)
           if images else [])
    zones, info = select_zones(owl, objects, tracks, cfg)
    return zones, info, owl


def select_zones(
    owl: Sequence[Sequence[Detection]], objects: Sequence[Sequence[Detection]], tracks: Sequence[Track],
    cfg: SceneConfig,
) -> tuple[dict[str, Optional[BBox]], dict[str, dict[str, object]]]:
    """Зоны по детекциям OWLv2 и COCO с учётом людей: рабочий стол — тот, у которого были люди
    (cfg.desk_by_people), сейф — по геометрии относительно стола, двери и роста человека
    (worker.perception.zones.infer_zones). Общая для пайплайна и scripts/rezone_raw.py."""
    people = [p.bbox for tr in tracks for p in tr.points]
    zones, info = infer_zones(owl, objects, min_score=cfg.zone_min_score,
                              person_boxes=people if cfg.desk_by_people else None,
                              person_height=track_height(tracks))
    return zones, info


def door_signal(frames: Sequence[Frame], persons: Sequence[Sequence[Detection]], door: Optional[BBox],
                cfg: SceneConfig) -> Optional[list[Optional[float]]]:
    """Расхождение полосы двери с её видом в начале ролика по каждому кадру (worker.perception.door).

    Людей закрашивают все рамки person детектора (и за стеклом тоже). Без моделей: кадры уже на диске.
    None — зоны двери нет (R15 молчит)."""
    if door is None or not frames:
        return None
    band = door_band(door, cfg.door_band_top, cfg.door_band_bottom)
    patches = []
    for frame, dets in zip(frames, persons):
        image = imread_unicode(frame.path)
        patches.append(None if image is None else
                       door_patch(image, band, [d.bbox_xyxy for d in dets], margin=cfg.door_person_margin))
    return door_diffs(patches, [f.timestamp_sec for f in frames],
                      baseline_sec=cfg.door_baseline_sec, max_masked=cfg.door_max_masked)


def scan_desk_items(frames: Sequence[Frame], bundle: Any, cfg: SceneConfig) -> list[tuple[float, list[Detection]]]:
    """OWLv2 по SCAN_QUERIES на кадрах раз в cfg.item_scan_sec (на длинном видео реже — не больше
    cfg.item_scan_max кадров): [(таймкод кадра, детекции)]."""
    if cfg.item_scan_sec <= 0 or not frames:
        return []
    # длинное видео — реже, чтобы скан укладывался в лимит задачи (не больше item_scan_max кадров)
    step = max(cfg.item_scan_sec, frames[-1].timestamp_sec / max(1, cfg.item_scan_max))
    picks: list[int] = []
    next_t = 0.0
    for pos, frame in enumerate(frames):
        if frame.timestamp_sec + 1e-6 >= next_t:
            picks.append(pos)
            next_t = frame.timestamp_sec + step
    out: list[tuple[float, list[Detection]]] = []
    for start in range(0, len(picks), 8):
        chunk = [(frames[k].timestamp_sec, imread_unicode(frames[k].path)) for k in picks[start:start + 8]]
        chunk = [(t, im) for t, im in chunk if im is not None]
        if chunk:
            dets = bundle.open_vocab([im for _, im in chunk], list(SCAN_QUERIES), score_threshold=cfg.zone_owl_score,
                                     top_k=3)
            out.extend((t, list(d)) for (t, _), d in zip(chunk, dets))
    return out


def check_left_items(timeline: SceneTimeline, frames: Sequence[Frame], bundle: Any, cfg: SceneConfig,
                     ) -> list[tuple[float, list[Detection]]]:
    """OWLv2 по предметам (ITEM_QUERIES) на кадрах после каждого ухода кассира (item_check_times):
    [(таймкод кадра, детекции)]. Без уходов — пусто, модель не вызывается."""
    times = item_check_times(timeline)
    if not times or not frames:
        return []
    stamps = [f.timestamp_sec for f in frames]
    picks = sorted({min(range(len(frames)), key=lambda k: abs(stamps[k] - t)) for t in times})
    images = [(frames[k].timestamp_sec, imread_unicode(frames[k].path)) for k in picks]
    images = [(t, im) for t, im in images if im is not None]
    if not images:
        return []
    dets = bundle.open_vocab([im for _, im in images], list(ITEM_QUERIES), score_threshold=cfg.zone_owl_score,
                             top_k=4)
    return [(t, list(d)) for (t, _), d in zip(images, dets)]


def person_inside_probs(frames: Sequence[Frame], tracks: Sequence[Track], zones: dict[str, Optional[BBox]],
                        bundle: Any, head: LinearHead, every_sec: float = 1.0) -> list[tuple[int, BBox, float]]:
    """Голова «внутри кассы» по точкам треков раз в every_sec на трек: [(кадр, рамка, p)].

    Вход головы — эмбеддинг SigLIP вырезки рамки (heads.crop_person) и геометрия heads.person_geo
    (рост — track_height, верх стола — zones["desk"]), как в scripts/embed_people.py при обучении.
    """
    import numpy as np

    ref_h = track_height(tracks)
    desk = zones.get("desk")
    picks: list[tuple[int, BBox, float]] = []
    for tr in tracks:
        next_t = -1.0
        for pt in tr.points:
            if pt.t + 1e-6 >= next_t:
                picks.append((pt.frame_index, pt.bbox, pt.score))
                next_t = pt.t + every_sec
    images: dict[int, Any] = {}
    keep, crops, geos = [], [], []
    for fi, box, score in picks:
        if fi not in images:
            images[fi] = imread_unicode(frames[fi].path)
        img = images[fi]
        crop = crop_person(img, box) if img is not None else None
        if crop is None:
            continue
        keep.append((fi, box))
        crops.append(crop)
        geos.append(person_geo(box, score, ref_h, desk))
    if not crops:
        return []
    feats = np.concatenate([bundle.zero_shot.image_embeddings(crops),
                            np.asarray(geos, dtype=np.float32) * np.asarray(PERSON_GEO_SCALE, dtype=np.float32)], axis=1)
    return [(fi, box, round(float(p), 4)) for (fi, box), p in zip(keep, head(feats))]


# Задачи дообученной башни, которые конвейер подставляет в сцену: задача -> зона вырезки.
# safe_open заменяет ряд state_samples.safe_open (голова/zero-shot), desk_cash — отдельное поле сцены
# desk_cash_ft (builder.with_ft_items) только для R14; desk_cash OWLv2-скана не меняется.
FT_FIELDS: dict[str, str] = {"safe_open": "safe", "desk_cash": "desk"}


def apply_safe_ft(samples: dict[str, list[tuple[float, Optional[float]]]], frames: Sequence[Frame],
                  zones: dict[str, Optional[BBox]], models_dir: Path, device: str, cfg: SceneConfig,
                  ) -> dict[str, Any]:
    """Поля из дообученной башни (SceneConfig.safe_ft, задачи — SceneConfig.safe_ft_fields) — на месте в samples.

    safe_open: вырезки те же, что у _sample_states (crop_zone зоны сейфа с ZONE_CROP_MARGIN на кадрах
    выборок safe_open); safe_open := вероятность башни, прежнее значение (голова раунда 3 или zero-shot)
    уходит в safe_open_head / safe_open_zero_shot.
    desk_cash (если башня его обучала): вырезка стола раз в cfg.state_every_sec — те же секунды, что
    у zero-shot стола, но без закрашивания людей (так башню учили, scripts/finetune_siglip.py);
    ряд samples["desk_cash"], в сцену (поле desk_cash_ft) его ставит builder.with_ft_items.
    Нет весов или ошибка инференса — samples не меняются (сцена как без флага), в meta — error.
    Возвращает meta.safe_ft (fields — какие поля пришли из башни).
    """
    tower, error = load_ft_tower(models_dir, cfg.safe_ft, device)
    if tower is None:
        return {"error": error or f"не загружена: {cfg.safe_ft}"}
    tasks = [t for t in cfg.safe_ft_fields if t in FT_FIELDS and t in tower.tasks]
    if not tasks:
        msg = f"в башне {tower.name} нет задач {list(cfg.safe_ft_fields)} (есть {sorted(tower.tasks)})"
        LOG.warning("%s: сцена считается без неё", msg)
        return {"error": msg}
    info = ft_tower_info(tower)
    by_ms = {int(round(f.timestamp_sec * 1000)): f for f in frames}

    def crops_at(times: Sequence[float], zone: str) -> tuple[list[float], list[Any]]:
        kept, crops = [], []
        for t in times:
            frame = by_ms.get(int(round(t * 1000)))
            image = imread_unicode(frame.path) if frame is not None else None
            crop = crop_zone(image, zones[zone], ZONE_CROP_MARGIN.get(zone, 0.1)) if image is not None else None
            if crop is not None:
                kept.append(t)
                crops.append(crop)
        return kept, crops

    wanted: dict[str, list[float]] = {}
    if "safe_open" in tasks and zones.get("safe") is not None and samples.get("safe_open"):
        wanted["safe_open"] = [t for t, _ in samples["safe_open"]]
    if "desk_cash" in tasks and zones.get("desk") is not None:
        wanted["desk_cash"] = [frames[pos].timestamp_sec for pos in _pick_positions(frames, cfg.state_every_sec)]
    out: dict[str, dict[int, float]] = {}
    try:
        for task, times in wanted.items():
            kept, crops = crops_at(times, FT_FIELDS[task])
            probs = tower(crops, [task], batch_size=cfg.state_batch_frames)[task] if crops else []
            out[task] = {int(round(t * 1000)): round(float(p), 4) for t, p in zip(kept, probs)}
    except Exception as exc:         # нехватка памяти и т.п.: сцена как без флага, задача не падает
        LOG.warning("дообученная башня %s: %s — сцена без неё", tower.name, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    if "safe_open" in out:
        current = samples["safe_open"]
        keep = "safe_open_head" if "safe_open_zero_shot" in samples else "safe_open_zero_shot"
        samples[keep] = list(current)
        samples["safe_open"] = [(t, out["safe_open"].get(int(round(t * 1000)))) for t, _ in current]
    if out.get("desk_cash"):
        samples["desk_cash"] = sorted((ms / 1000.0, p) for ms, p in out["desk_cash"].items())
    fields = [t for t in tasks if t == "safe_open" or out.get(t)]
    return {**info, "fields": fields, "samples": {k: len(v) for k, v in out.items()}}


def _pick_positions(frames: Sequence[Frame], every_sec: float) -> list[int]:
    """Позиции кадров раз в every_sec (первый кадр не раньше предыдущего + every_sec) — как в _sample_states."""
    picks: list[int] = []
    next_t = 0.0
    for pos, frame in enumerate(frames):
        if frame.timestamp_sec + 1e-6 >= next_t:
            picks.append(pos)
            next_t = frame.timestamp_sec + every_sec
    return picks


def _mask_people(image: Any, people: Sequence[Detection]) -> Any:
    """Копия кадра, где рамки людей закрашены серым (люди не попадают в zero-shot-оценку стола)."""
    out = image.copy()
    h, w = out.shape[:2]
    for det in people:
        x1, y1, x2, y2 = det.bbox_xyxy
        out[int(y1 * h):int(y2 * h) + 1, int(x1 * w):int(x2 * w) + 1] = 128
    return out


def _sample_states(
    frames: Sequence[Frame], zones: dict[str, Optional[BBox]], bundle: Any, cfg: SceneConfig,
    persons: Optional[Sequence[Sequence[Detection]]] = None,
    *,
    safe_head: Optional[LinearHead] = None,
    head_errors: Optional[list[str]] = None,
) -> dict[str, list[tuple[float, Optional[float]]]]:
    """Zero-shot-оценки состояний по вырезкам зон раз в cfg.state_every_sec: {состояние: [(t, p)]}.

    persons — рамки людей по кадрам; при cfg.mask_people_in_desk они закрашиваются в вырезке стола.
    safe_head — голова «сейф открыт»: её вероятность становится safe_open, zero-shot — safe_open_zero_shot.
    Ошибка инференса головы пишется в head_errors (если передан список), дальше — zero-shot.
    """
    if head_errors is None:
        head_errors = []
    states = enabled_states(cfg)
    by_zone: dict[str, list[str]] = {}
    for name, spec in states.items():
        if zones.get(spec.zone) is not None:
            by_zone.setdefault(spec.zone, []).append(name)
    if not by_zone:
        return {}

    picks = _pick_positions(frames, cfg.state_every_sec)

    samples: dict[str, list[tuple[float, Optional[float]]]] = {n: [] for names in by_zone.values() for n in names}
    for start in range(0, len(picks), cfg.state_batch_frames):
        chunk = picks[start:start + cfg.state_batch_frames]
        images = {pos: imread_unicode(frames[pos].path) for pos in chunk}
        for zone, names in by_zone.items():
            crops: list[Any] = []
            times: list[float] = []
            for pos in chunk:
                image = images[pos]
                if image is not None and zone == "desk" and cfg.mask_people_in_desk and persons is not None:
                    image = _mask_people(image, persons[pos])
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
            if zone == "safe" and safe_head is not None and "safe_open" in names:
                try:
                    head_p = safe_head(bundle.zero_shot.image_embeddings(crops))
                except Exception as exc:     # не та размерность весов и т.п.: дальше zero-shot, задача не падает
                    LOG.warning("голова %s: %s — safe_open остаётся zero-shot", safe_head.name, exc)
                    head_errors.append(str(exc))
                    safe_head = None
                    done = samples.pop("safe_open_zero_shot", [])     # ряд целиком zero-shot, без смеси
                    samples["safe_open"][:len(done)] = done
                    continue
                samples.setdefault("safe_open_zero_shot", []).extend(samples["safe_open"][-len(crops):])
                samples["safe_open"][-len(crops):] = [(t, round(float(p), 4)) for t, p in zip(times, head_p)]
    return samples
