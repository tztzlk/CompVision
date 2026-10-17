"""Зоны сцены (стол, сейф, дверь, ноутбук) по open-vocabulary детекциям на ключевых кадрах.

Чистая логика без моделей: на вход — детекции OWLv2 (label = текст запроса) по нескольким
ключевым кадрам и COCO-детекции RT-DETR по всем кадрам (запасной источник стола и ноутбука).
Зона рабочего места (workplace) считается позже, по треку кассира (worker.scene.builder).
Ничего не привязано к координатам конкретной съёмки: финальное видео снято под другим углом.
"""

from __future__ import annotations

from typing import Optional, Sequence

from worker.perception import Detection
from worker.perception.geometry import BBox, ioa, iou, median_box
from worker.perception.states import TABLE_LABELS

__all__ = ["ZONE_QUERIES", "COCO_FALLBACK", "all_zone_queries", "infer_zones", "desk_candidates", "pick_desk_by_people"]

# Текстовые запросы OWLv2 для каждой зоны (английский — язык обучения модели).
ZONE_QUERIES: dict[str, tuple[str, ...]] = {
    "desk": ("a wooden office desk", "a desk with a computer"),
    "safe": ("a metal safe", "a grey metal cabinet"),
    "door": ("a glass door", "a door"),
    "laptop": ("a laptop",),
}

# Если OWLv2 зону не нашёл — классы COCO из RT-DETR.
COCO_FALLBACK: dict[str, tuple[str, ...]] = {
    "desk": TABLE_LABELS,
    "laptop": ("laptop",),
}

# В кадрах кейса на переднем плане стоит большой белый стол, обрезанный нижним краем кадра.
# Кандидат «стол», касающийся нижнего края, получает штраф, чтобы выигрывал рабочий стол кассира.
_BOTTOM_EDGE = 0.98
_BOTTOM_EDGE_PENALTY = 0.5
_CLUSTER_IOU = 0.5
# Замер 17.10 по сырому кэшу 42 видео: сумма score по группе выбирала то, что совпало с обоими
# запросами зоны, а не то, что лучше всего похоже на неё. Стол за стеклом соседнего офиса
# («wooden desk» 0,40 + «desk with a computer» 0,35) обходил рабочий стол (0,54) на Х-20…Х-29,
# белая тумба — серый сейф на Х-1…Х-5, Х-12…Х-15. Теперь группа оценивается так:
#   desk — сумма по ключевым кадрам лучшего score кадра (запросы не складываются) и рамка не
#          мельче _MIN_AREA (стол за стеклом — 0,5 % кадра, рабочий — ~12 %);
#   safe, laptop — лучший одиночный score;
#   door — как было (сумма): выбор двери по OWLv2 ненадёжен, вход ищется по трекам.
_SCORING = {"desk": "keyframe_max", "safe": "max", "laptop": "max", "door": "sum"}
_MIN_AREA = {"desk": 0.03}


def all_zone_queries() -> list[str]:
    """Все запросы OWLv2 без повторов (порядок стабилен)."""
    return list(dict.fromkeys(q for queries in ZONE_QUERIES.values() for q in queries))


def _penalty(zone: str, box: BBox) -> float:
    if zone == "desk" and box[3] >= _BOTTOM_EDGE:
        return _BOTTOM_EDGE_PENALTY
    return 1.0


def _clusters(candidates: Sequence[tuple[BBox, float]]) -> list[list[tuple[BBox, float]]]:
    """Сгруппировать похожие рамки (IoU >= 0.5 с первой рамкой группы)."""
    clusters: list[list[tuple[BBox, float]]] = []
    for box, score in sorted(candidates, key=lambda c: c[1], reverse=True):
        for cluster in clusters:
            if iou(cluster[0][0], box) >= _CLUSTER_IOU:
                cluster.append((box, score))
                break
        else:
            clusters.append([(box, score)])
    return clusters


def desk_candidates(owl_per_keyframe: Sequence[Sequence[Detection]], min_score: float = 0.12) -> list[tuple[BBox, float]]:
    """Все группы-кандидаты «стол» по OWLv2: (медианная рамка, сумма score со штрафом), лучшие первыми."""
    candidates = [
        (det.bbox_xyxy, det.score * _penalty("desk", det.bbox_xyxy))
        for dets in owl_per_keyframe
        for det in dets
        if det.label in ZONE_QUERIES["desk"] and det.score >= min_score
    ]
    out = [(median_box(b for b, _ in cl), sum(s for _, s in cl)) for cl in _clusters(candidates)]
    return sorted(((b, s) for b, s in out if b is not None), key=lambda c: -c[1])


def pick_desk_by_people(candidates: Sequence[tuple[BBox, float]], person_boxes: Sequence[BBox],
                        margin: float = 0.15, min_share: float = 0.2) -> Optional[BBox]:
    """Рабочий стол — тот, у которого больше всего сидели и стояли люди, а не самый заметный
    (на переднем плане кадров кейса большой пустой белый стол). Для каждого кандидата —
    сумма долей рамок людей в расширенной рамке стола по всем кадрам; кандидат с долей меньше
    min_share от лучшего по OWLv2 не рассматривается. None — людей нет или кандидат один."""
    if len(candidates) < 2 or not person_boxes:
        return None
    top = candidates[0][1]
    pool = [(b, s) for b, s in candidates if s >= min_share * top]

    def presence(box: BBox) -> float:
        zone = _expand_rel(box, margin)
        return sum(ioa(p, zone) for p in person_boxes)

    best = max(pool, key=lambda c: presence(c[0]))
    return best[0] if presence(best[0]) > 0 else None


def _expand_rel(box: BBox, frac: float) -> BBox:
    dx, dy = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return (max(0.0, box[0] - dx), max(0.0, box[1] - dy), min(1.0, box[2] + dx), min(1.0, box[3] + dy))


def _scored_cluster(candidates: Sequence[tuple[BBox, float, int]], scoring: str) -> tuple[Optional[BBox], float]:
    """Группа похожих рамок с наибольшим счётом: sum — сумма score; keyframe_max — сумма по ключевым
    кадрам лучшего score кадра; max — лучший одиночный score. Возвращает (медианная рамка, сумма score)."""
    clusters = _clusters([(b, s) for b, s, _ in candidates])
    if not clusters:
        return None, 0.0
    frame_of = {}
    for b, s, k in candidates:
        frame_of.setdefault((tuple(b), s), k)

    def score(cl: list[tuple[BBox, float]]) -> float:
        if scoring == "max":
            return max(s for _, s in cl)
        if scoring == "keyframe_max":
            best: dict[int, float] = {}
            for b, s in cl:
                k = frame_of[(tuple(b), s)]
                best[k] = max(best.get(k, 0.0), s)
            return sum(best.values())
        return sum(s for _, s in cl)

    top = max(clusters, key=score)
    return median_box(b for b, _ in top), sum(s for _, s in top)


def _best_cluster(candidates: Sequence[tuple[BBox, float]]) -> tuple[Optional[BBox], float]:
    """Сгруппировать похожие рамки (IoU >= 0.5) и выбрать группу с наибольшей суммой score."""
    clusters: list[list[tuple[BBox, float]]] = []
    for box, score in sorted(candidates, key=lambda c: c[1], reverse=True):
        for cluster in clusters:
            if iou(cluster[0][0], box) >= _CLUSTER_IOU:
                cluster.append((box, score))
                break
        else:
            clusters.append([(box, score)])
    if not clusters:
        return None, 0.0
    best = max(clusters, key=lambda cl: sum(s for _, s in cl))
    return median_box(b for b, _ in best), sum(s for _, s in best)


def infer_zones(
    owl_per_keyframe: Sequence[Sequence[Detection]],
    coco_per_frame: Sequence[Sequence[Detection]],
    *,
    min_score: float = 0.12,
    coco_min_score: float = 0.5,
    coco_min_share: float = 0.2,
) -> tuple[dict[str, Optional[BBox]], dict[str, dict[str, object]]]:
    """Зоны и сведения о том, откуда они взяты: ({зона: рамка | None}, {зона: {source, score}})."""
    keyframes = max(1, len(owl_per_keyframe))
    zones: dict[str, Optional[BBox]] = {}
    info: dict[str, dict[str, object]] = {}
    for zone, queries in ZONE_QUERIES.items():
        candidates = [
            (det.bbox_xyxy, det.score * _penalty(zone, det.bbox_xyxy), k)
            for k, dets in enumerate(owl_per_keyframe)
            for det in dets
            if det.label in queries and det.score >= min_score
            and (det.bbox_xyxy[2] - det.bbox_xyxy[0]) * (det.bbox_xyxy[3] - det.bbox_xyxy[1]) >= _MIN_AREA.get(zone, 0.0)
        ]
        box, total = _scored_cluster(candidates, _SCORING.get(zone, "sum"))
        if box is not None:
            zones[zone] = box
            info[zone] = {"source": "owlv2", "score": round(min(1.0, total / keyframes), 3)}
            continue

        labels = COCO_FALLBACK.get(zone, ())
        best_per_frame = []
        for dets in coco_per_frame:
            hits = [d for d in dets if d.label in labels and d.score >= coco_min_score]
            if hits:
                best_per_frame.append(max(hits, key=lambda d: d.score).bbox_xyxy)
        if labels and best_per_frame and len(best_per_frame) >= coco_min_share * max(1, len(coco_per_frame)):
            zones[zone] = median_box(best_per_frame)
            info[zone] = {"source": "coco", "score": round(len(best_per_frame) / max(1, len(coco_per_frame)), 3)}
        else:
            zones[zone] = None
            info[zone] = {"source": None, "score": 0.0}
    return zones, info
