"""Повторная идентификация кассира по эмбеддингам вырезок (SceneConfig.cashier_reid, только stdlib).

Трекер теряет кассира и заводит новый трек (Х-35: сидящая кассир тянется к сейфу — на её месте
трек 7 начинается раньше, чем кончился трек 1; Х-32: смена трека в двери). stitch_tracks такие
пары не склеивает: разрыв отрицательный. Здесь — после выбора кассира: трек кассира оборвался,
а не позже чем через reid_gap_sec (или с перекрытием до reid_overlap_sec) у двери или на его месте
появился трек, чей эмбеддинг первых точек похож на эмбеддинг последних точек кассира
(косинус >= reid_min_cos), — это тот же кассир, треки склеиваются.

Эмбеддинги — person_reid сырого кэша (scripts/add_reid_to_raw.py): [[id исходного трека,
средний эмбеддинг первых точек, средний эмбеддинг последних точек], ...] (SigLIP, нормированные).
"""

from __future__ import annotations

import math
from typing import Any, Collection, Optional, Sequence

from worker.perception.geometry import BBox, center, center_dist, contains_point, ioa, pad
from worker.perception.tracking import Track

__all__ = ["cosine", "reid_stitch"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _key(track_point: Any) -> tuple:
    return (int(track_point.frame_index), *(round(float(v), 4) for v in track_point.bbox))


def _near_door(box: BBox, door_zone: Optional[BBox], min_ioa: float) -> bool:
    if door_zone is None:
        return False
    cx, cy = center(box)
    return contains_point(door_zone, cx, cy) or ioa(box, door_zone) >= min_ioa


def reid_stitch(raw_tracks: Sequence[Track], stitched: Sequence[Track], cashier_ids: Collection[int],
                workplace: Optional[BBox], door: Optional[BBox], person_reid: Sequence[Sequence[Any]],
                config: Any) -> Optional[list[Track]]:
    """Склеить треки кассира с продолжениями по эмбеддингу. None — склеивать нечего (сцена как без reid).

    raw_tracks — исходные треки (по ним эмбеддинги person_reid), stitched — после stitch_tracks;
    cashier_ids — id треков с ролью кассира. Склеенный трек: точки кассира до начала продолжения
    и все точки продолжения, id — кассира.
    """
    emb = {int(row[0]): (row[1], row[2]) for row in person_reid}
    first_of = {_key(t.points[0]): t.track_id for t in raw_tracks if t.points}
    last_of = {_key(t.points[-1]): t.track_id for t in raw_tracks if t.points}

    def first_emb(t: Track) -> Optional[Sequence[float]]:
        rid = first_of.get(_key(t.points[0]))
        return emb[rid][0] if rid in emb else None

    def last_emb(t: Track) -> Optional[Sequence[float]]:
        rid = last_of.get(_key(t.points[-1]))
        return emb[rid][1] if rid in emb else None

    door_zone = pad(door, config.door_margin) if door else None
    tracks = [Track(t.track_id, list(t.points)) for t in stitched]
    cashier = set(cashier_ids)
    changed = False
    while True:
        best: Optional[tuple[float, Track, Track]] = None
        for a in tracks:
            if a.track_id not in cashier or not a.points:
                continue
            ea = last_emb(a)
            if ea is None:
                continue
            for b in tracks:
                if b is a or not b.points or b.last_t <= a.last_t or b.first_t <= a.first_t:
                    continue
                gap = b.first_t - a.last_t
                if gap > config.reid_gap_sec or gap < -config.reid_overlap_sec:
                    continue
                start = b.points[0].bbox
                at_place = (_near_door(start, door_zone, config.door_min_ioa)
                            or (workplace is not None and ioa(start, workplace) >= config.cashier_extra_ioa)
                            or center_dist(a.points[-1].bbox, start) <= config.stitch_center_dist)
                if not at_place:
                    continue
                eb = first_emb(b)
                if eb is None:
                    continue
                cos = cosine(ea, eb)
                if cos >= config.reid_min_cos and (best is None or cos > best[0]):
                    best = (cos, a, b)
        if best is None:
            break
        _, a, b = best
        a.points[:] = [p for p in a.points if p.t < b.first_t] + list(b.points)
        tracks.remove(b)
        changed = True
    return tracks if changed else None
