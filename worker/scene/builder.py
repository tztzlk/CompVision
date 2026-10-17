"""Сборка SceneTimeline из треков, детекций, зон и zero-shot-оценок (чистая логика, только stdlib).

Модели запускает worker.scene.pipeline; здесь — правила восприятия сцены:
  - склейка разорванных треков и отбрасывание коротких;
  - «внутри кассы / за стеклом» по высоте рамки и положению относительно стола;
  - роль кассира без распознавания лиц: трек, дольше всех и ближе всех находящийся у стола;
  - зона рабочего места (workplace) по положениям кассира у стола;
  - вход и выход через зону двери;
  - состояния предметов: классы COCO у рук кассира и на столе + zero-shot-оценки по зонам.
Все пороги — в SceneConfig; их подбирают на машине с GPU по контактным листам scripts/viz_scene.py.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Mapping, Optional, Sequence

from worker.perception import Detection
from worker.perception.geometry import (
    BBox,
    area,
    center,
    center_dist,
    contains_point,
    expand,
    height,
    ioa,
    median_box,
    pad,
)
from worker.perception.states import BAG_LABELS, FOOD_LABELS, PHONE_LABELS
from worker.perception.tracking import Track, TrackPoint
from worker.scene.contract import ZONE_NAMES, Person, SceneState, SceneTimeline

__all__ = ["SceneConfig", "FrameRef", "stitch_tracks", "assemble_timeline"]


@dataclass(frozen=True)
class SceneConfig:
    """Параметры восприятия сцены (доли кадра, секунды, вероятности)."""

    # кадры и детектор
    fps: float = 3.0                    # частота анализа кадров
    batch_size: int = 8                 # кадров на один прогон RT-DETR
    jpeg_quality: int = 90
    person_score: float = 0.35          # минимальный score человека
    object_score: float = 0.30          # минимальный score предмета COCO
    person_nms_iou: float = 0.7         # подавление дублей рамок человека
    # трекер
    track_iou: float = 0.2
    track_center_dist: float = 0.12
    track_max_lost_sec: float = 1.5
    track_new_score: float = 0.5
    stitch_gap_sec: float = 1.0         # трек, пропавший не дольше, склеивается со следующим
    stitch_center_dist: float = 0.15
    min_track_points: int = 2
    # внутри кассы / за стеклом
    inside_height_ratio: float = 0.5    # высота рамки >= доли типичной высоты человека в кассе
    floor_margin: float = 0.0           # низ рамки ниже верхнего края стола (+ запас) — человек в комнате
    hold_gap_sec: float = 1.0           # пропуск детекции внутри трека не дольше — человек на месте
    # кассир и рабочее место
    desk_margin: float = 0.15           # расширение зоны стола при поиске кассира и предметов
    workplace_min_desk_ioa: float = 0.15
    workplace_margin: float = 0.05
    presence_ioa: float = 0.5           # доля рамки кассира в рабочей зоне, при которой presence = 1
    cashier_extra_ioa: float = 0.4      # трек вне времени основного кассира на его месте — тоже кассир
    # вход / выход
    door_margin: float = 0.08
    door_min_ioa: float = 0.1
    entry_grace_sec: float = 0.7        # кто в кадре с самого начала — не «вошёл»
    exit_grace_sec: float = 0.7
    min_event_track_sec: float = 0.6    # трек внутри короче — не порождает вход/выход
    # предметы
    hand_margin: float = 0.15           # расширение рамки кассира: «предмет в руке»
    # зоны и zero-shot
    zone_keyframes: int = 4
    zone_min_score: float = 0.12
    zone_top_k: int = 3
    state_every_sec: float = 1.0
    state_batch_frames: int = 16
    enable_door_state: bool = False
    enable_collection_bag: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameRef:
    index: int          # позиция кадра в анализе (совпадает с TrackPoint.frame_index)
    t: float
    frame: str          # JPEG относительно work_dir, например "frames/000051.jpg"


@dataclass(frozen=True)
class _Entry:
    bbox: BBox
    inside: bool


@dataclass
class _TrackInfo:
    track: Track
    inside: list[bool]
    by_frame: dict[int, _Entry] = field(default_factory=dict)
    role: str = "other"                 # "cashier" | "other"

    def inside_points(self) -> list[TrackPoint]:
        return [p for p, ins in zip(self.track.points, self.inside) if ins]


# --------------------------------------------------------------------------- треки


def stitch_tracks(tracks: Sequence[Track], config: SceneConfig) -> list[Track]:
    """Склеить трек с более ранним, если тот пропал не дольше stitch_gap_sec назад рядом.

    Возвращает новые объекты Track (входные не меняются); id склеенного — id более раннего.
    """
    ordered = sorted((Track(t.track_id, list(t.points)) for t in tracks if t.points), key=lambda tr: tr.first_t)
    merged: list[Track] = []
    for track in ordered:
        best: Optional[Track] = None
        best_dist = math.inf
        for candidate in merged:
            gap = track.first_t - candidate.last_t
            if gap <= 0 or gap > config.stitch_gap_sec:
                continue
            dist = center_dist(candidate.points[-1].bbox, track.points[0].bbox)
            if dist <= config.stitch_center_dist and dist < best_dist:
                best, best_dist = candidate, dist
        if best is None:
            merged.append(track)
        else:
            best.points.extend(track.points)
    return merged


def _reference_height(tracks: Sequence[Track]) -> float:
    """Типичная высота человека внутри кассы: наибольшая медианная высота среди долгих треков."""
    long_tracks = [t for t in tracks if t.duration >= 1.0] or list(tracks)
    if not long_tracks:
        return 0.0
    return max(median(height(p.bbox) for p in t.points) for t in long_tracks)


def _inside_flags(track: Track, ref_height: float, desk: Optional[BBox], config: SceneConfig) -> list[bool]:
    """Внутри кассы ли человек в каждой точке трека (сглажено большинством по 3 точкам).

    Люди за стеклом и отражения — мелкие рамки, низ которых выше верхнего края стола.
    """
    raw = []
    for point in track.points:
        by_height = ref_height > 0 and height(point.bbox) >= config.inside_height_ratio * ref_height
        by_floor = desk is not None and point.bbox[3] >= desk[1] + config.floor_margin
        raw.append(by_height or by_floor)
    smoothed = []
    for i, value in enumerate(raw):
        window = raw[max(0, i - 1): i + 2]
        votes = sum(window) * 2
        smoothed.append(value if votes == len(window) else votes > len(window))
    return smoothed


def _fill_frames(info: _TrackInfo, times: Sequence[float], config: SceneConfig) -> None:
    """Сопоставить трек кадрам; короткие пропуски детекции (<= hold_gap_sec) заполнить ближайшей точкой."""
    points = info.track.points
    for point, inside in zip(points, info.inside):
        info.by_frame[point.frame_index] = _Entry(point.bbox, inside)
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if b.t - a.t > config.hold_gap_sec:
            continue
        for k in range(a.frame_index + 1, b.frame_index):
            nearer_a = times[k] - a.t <= b.t - times[k]
            source, inside = (a, info.inside[i]) if nearer_a else (b, info.inside[i + 1])
            info.by_frame[k] = _Entry(source.bbox, inside)


# --------------------------------------------------------------------------- роли и события


def _select_cashier(infos: Sequence[_TrackInfo], desk: Optional[BBox], config: SceneConfig) -> Optional[_TrackInfo]:
    """Кассир — трек с наибольшей суммой «близость к столу × размер» по точкам внутри кассы.

    Размер нужен, когда у стола двое: кассир ближе к камере, чем клиент напротив.
    """
    desk_zone = expand(desk, config.desk_margin) if desk else None
    best: Optional[_TrackInfo] = None
    best_score = 0.0
    for info in infos:
        score = 0.0
        for point, inside in zip(info.track.points, info.inside):
            if not inside:
                continue
            weight = math.sqrt(area(point.bbox))
            if desk_zone is not None:
                weight *= 0.25 + ioa(point.bbox, desk_zone)
            score += weight
        if score > best_score:
            best, best_score = info, score
    return best


def _workplace(cashier: _TrackInfo, desk: Optional[BBox], config: SceneConfig) -> Optional[BBox]:
    """Зона рабочего места: медиана рамок кассира у стола (или всех его рамок, если стол не найден)."""
    boxes = [p.bbox for p in cashier.inside_points()]
    if not boxes:
        return None
    if desk is not None:
        desk_zone = expand(desk, config.desk_margin)
        near = [b for b in boxes if ioa(b, desk_zone) >= config.workplace_min_desk_ioa]
        if near:
            boxes = near
    box = median_box(boxes)
    return expand(box, config.workplace_margin) if box else None


def _mark_extra_cashiers(infos: Sequence[_TrackInfo], cashier: _TrackInfo, workplace: Optional[BBox],
                         config: SceneConfig) -> None:
    """Трек, не пересекающийся по времени с кассиром и сидящий на его месте, — тот же кассир."""
    if workplace is None:
        return
    main = cashier.track
    for info in infos:
        points = info.inside_points()
        if info.role == "cashier" or not points:
            continue
        if not (points[-1].t < main.first_t or points[0].t > main.last_t):
            continue
        if median(ioa(p.bbox, workplace) for p in points) >= config.cashier_extra_ioa:
            info.role = "cashier"


def _near_door(box: BBox, door_zone: Optional[BBox], config: SceneConfig) -> bool:
    if door_zone is None:
        return True        # дверь не найдена: вход/выход — по появлению и исчезновению трека
    cx, cy = center(box)
    return contains_point(door_zone, cx, cy) or ioa(box, door_zone) >= config.door_min_ioa


def _door_events(infos: Sequence[_TrackInfo], door: Optional[BBox], duration: float,
                 config: SceneConfig) -> tuple[set[int], set[int]]:
    """Позиции кадров, где кто-то вошёл (первая точка внутри у двери) и вышел (последняя у двери)."""
    door_zone = pad(door, config.door_margin) if door else None
    entered: set[int] = set()
    exited: set[int] = set()
    for info in infos:
        points = info.inside_points()
        if not points or points[-1].t - points[0].t < config.min_event_track_sec:
            continue
        first, last = points[0], points[-1]
        if first.t > config.entry_grace_sec and _near_door(first.bbox, door_zone, config):
            entered.add(first.frame_index)
        if last.t < duration - config.exit_grace_sec and _near_door(last.bbox, door_zone, config):
            exited.add(last.frame_index)
    return entered, exited


# --------------------------------------------------------------------------- предметы


def _max_near(dets: Sequence[Detection], labels: Sequence[str], zone: BBox, min_score: float) -> float:
    """Наибольший score предмета из labels, центр которого внутри zone (0, если таких нет)."""
    best = 0.0
    for det in dets:
        if det.label in labels and det.score >= min_score:
            cx, cy = center(det.bbox_xyxy)
            if contains_point(zone, cx, cy):
                best = max(best, det.score)
    return best


def _max_opt(*values: Optional[float]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _median3(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Медиана по окну из 3 точек; None остаётся None и не участвует в соседних окнах."""
    out: list[Optional[float]] = []
    for i, value in enumerate(values):
        if value is None:
            out.append(None)
            continue
        window = [v for v in values[max(0, i - 1): i + 2] if v is not None]
        out.append(median(window))
    return out


class _Held:
    """Значение zero-shot-оценки в момент t: последняя оценка не позже t (до первой — первая)."""

    def __init__(self, samples: Sequence[tuple[float, Optional[float]]]) -> None:
        ordered = sorted(samples, key=lambda s: s[0])
        self._times = [t for t, _ in ordered]
        self._values = [v for _, v in ordered]

    def at(self, t: float) -> Optional[float]:
        if not self._times:
            return None
        i = bisect_right(self._times, t + 1e-6) - 1
        return self._values[max(i, 0)]


# --------------------------------------------------------------------------- сборка


def assemble_timeline(
    *,
    video_id: str,
    duration_sec: float,
    frame_size: tuple[int, int],
    frames: Sequence[FrameRef],
    tracks: Sequence[Track],
    objects: Sequence[Sequence[Detection]],
    zones: Mapping[str, Optional[BBox]],
    state_samples: Mapping[str, Sequence[tuple[float, Optional[float]]]],
    config: Optional[SceneConfig] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> SceneTimeline:
    """Собрать SceneTimeline.

    frames — проанализированные кадры по порядку; objects — предметы COCO (не люди) по одному
    списку на кадр; tracks — треки людей (frame_index = позиция кадра во frames); zones — стол,
    сейф, дверь, ноутбук (workplace считается здесь); state_samples — zero-shot-оценки
    {состояние: [(t, вероятность)]}, между оценками держится последнее значение.
    """
    cfg = config or SceneConfig()
    if len(objects) != len(frames):
        raise ValueError("objects: ожидается по одному списку детекций на кадр")
    times = [f.t for f in frames]
    duration = max(float(duration_sec), times[-1] if times else 0.0)
    desk, door = zones.get("desk"), zones.get("door")
    warnings: list[str] = []
    if desk is None:
        warnings.append("стол не найден: кассир выбран по размеру и длительности трека, предметы на столе = null")
    if door is None:
        warnings.append("дверь не найдена: вход и выход определяются по появлению и исчезновению трека")

    stitched = [t for t in stitch_tracks(tracks, cfg) if len(t.points) >= cfg.min_track_points]
    ref_height = _reference_height(stitched)
    infos = [_TrackInfo(track=t, inside=_inside_flags(t, ref_height, desk, cfg)) for t in stitched]
    for info in infos:
        _fill_frames(info, times, cfg)

    cashier = _select_cashier(infos, desk, cfg)
    workplace: Optional[BBox] = None
    if cashier is None:
        warnings.append("кассир не определён: cashier_present = null")
    else:
        cashier.role = "cashier"
        workplace = _workplace(cashier, desk, cfg)
        _mark_extra_cashiers(infos, cashier, workplace, cfg)
    entered, exited = _door_events(infos, door, duration, cfg)

    raw_presence: list[Optional[float]] = []
    per_frame: list[tuple[int, Optional[BBox], tuple[Person, ...]]] = []
    for pos in range(len(frames)):
        inside = [(info, info.by_frame[pos]) for info in infos if pos in info.by_frame and info.by_frame[pos].inside]
        cashier_entries = [entry for info, entry in inside if info.role == "cashier"]
        cashier_bbox: Optional[BBox] = None
        presence: Optional[float]
        if cashier is None:
            presence = None
        elif cashier_entries:
            if workplace is not None:
                cashier_bbox = max(cashier_entries, key=lambda e: ioa(e.bbox, workplace)).bbox
                presence = min(1.0, ioa(cashier_bbox, workplace) / cfg.presence_ioa)
            else:
                cashier_bbox = max(cashier_entries, key=lambda e: area(e.bbox)).bbox
                presence = 1.0
        else:
            presence = 0.0
        raw_presence.append(presence)
        others = tuple(
            Person(track_id=info.track.track_id, bbox=entry.bbox, first_seen_t=round(info.inside_points()[0].t, 3))
            for info, entry in inside if info.role != "cashier"
        )
        per_frame.append((len(inside), cashier_bbox, others))

    held = {name: _Held(samples) for name, samples in state_samples.items()}
    desk_zone = expand(desk, cfg.desk_margin) if desk else None
    states: list[SceneState] = []
    for pos, (ref, presence) in enumerate(zip(frames, _median3(raw_presence))):
        persons, cashier_bbox, others = per_frame[pos]
        dets = objects[pos]
        hand_zone = expand(cashier_bbox, cfg.hand_margin) if cashier_bbox else None
        zero_shot = {name: h.at(ref.t) for name, h in held.items()}
        states.append(SceneState(
            t=round(ref.t, 3),
            frame=ref.frame,
            persons_in_room=persons,
            cashier_present=None if presence is None else round(presence, 3),
            cashier_bbox=cashier_bbox,
            others=others,
            person_entered=pos in entered,
            person_exited=pos in exited,
            safe_open=zero_shot.get("safe_open"),
            door_open=zero_shot.get("door_open"),
            cash_on_desk=zero_shot.get("cash_on_desk"),
            bag_on_desk=_max_opt(
                _max_near(dets, BAG_LABELS, desk_zone, cfg.object_score) if desk_zone else None,
                zero_shot.get("bag_on_desk"),
            ),
            collection_bag=zero_shot.get("collection_bag"),
            outerwear_on_desk=zero_shot.get("outerwear_on_desk"),
            cup_or_food=_max_opt(
                _max_near(dets, FOOD_LABELS, hand_zone, cfg.object_score) if hand_zone else None,
                _max_near(dets, FOOD_LABELS, desk_zone, cfg.object_score) if desk_zone else None,
            ),
            phone_in_hand=_max_near(dets, PHONE_LABELS, hand_zone, cfg.object_score) if hand_zone else None,
            laptop_unlocked=zero_shot.get("laptop_unlocked"),
        ))

    out_zones: dict[str, Optional[BBox]] = {name: zones.get(name) for name in ZONE_NAMES}
    out_zones["workplace"] = workplace
    out_zones.update({name: box for name, box in zones.items() if name not in out_zones})
    full_meta = dict(meta or {})
    full_meta["scene"] = {
        "config": cfg.as_dict(),
        "reference_height": round(ref_height, 4),
        "tracks_raw": len(tracks),
        "tracks": len(stitched),
        "tracks_inside": sum(1 for info in infos if info.inside_points()),
        "cashier_track_ids": sorted(info.track.track_id for info in infos if info.role == "cashier"),
        "entered_t": [round(times[p], 3) for p in sorted(entered)],
        "exited_t": [round(times[p], 3) for p in sorted(exited)],
        "warnings": warnings,
    }
    return SceneTimeline(
        video_id=video_id,
        duration_sec=round(duration, 3),
        fps_sampled=cfg.fps,
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        zones=out_zones,
        states=tuple(states),
        meta=full_meta,
    )
