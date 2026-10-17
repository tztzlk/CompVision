"""Зоны сцены (стол, сейф, дверь, ноутбук) по open-vocabulary детекциям на ключевых кадрах.

Чистая логика без моделей: на вход — детекции OWLv2 (label = текст запроса) по нескольким
ключевым кадрам и COCO-детекции RT-DETR по всем кадрам (запасной источник стола и ноутбука).
Зона рабочего места (workplace) считается позже, по треку кассира (worker.scene.builder).
Ничего не привязано к координатам конкретной съёмки: финальное видео снято под другим углом.
"""

from __future__ import annotations

from typing import Optional, Sequence

from worker.perception import Detection
from worker.perception.geometry import BBox, iou, median_box

__all__ = ["ZONE_QUERIES", "COCO_FALLBACK", "all_zone_queries", "infer_zones"]

# Текстовые запросы OWLv2 для каждой зоны (английский — язык обучения модели).
ZONE_QUERIES: dict[str, tuple[str, ...]] = {
    "desk": ("a wooden office desk", "a desk with a computer"),
    "safe": ("a metal safe", "a grey metal cabinet"),
    "door": ("a glass door", "a door"),
    "laptop": ("a laptop",),
}

# Если OWLv2 зону не нашёл — классы COCO из RT-DETR.
COCO_FALLBACK: dict[str, tuple[str, ...]] = {
    "desk": ("dining table",),
    "laptop": ("laptop",),
}

# В кадрах кейса на переднем плане стоит большой белый стол, обрезанный нижним краем кадра.
# Кандидат «стол», касающийся нижнего края, получает штраф, чтобы выигрывал рабочий стол кассира.
_BOTTOM_EDGE = 0.98
_BOTTOM_EDGE_PENALTY = 0.5
_CLUSTER_IOU = 0.5


def all_zone_queries() -> list[str]:
    """Все запросы OWLv2 без повторов (порядок стабилен)."""
    return list(dict.fromkeys(q for queries in ZONE_QUERIES.values() for q in queries))


def _penalty(zone: str, box: BBox) -> float:
    if zone == "desk" and box[3] >= _BOTTOM_EDGE:
        return _BOTTOM_EDGE_PENALTY
    return 1.0


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
            (det.bbox_xyxy, det.score * _penalty(zone, det.bbox_xyxy))
            for dets in owl_per_keyframe
            for det in dets
            if det.label in queries and det.score >= min_score
        ]
        box, total = _best_cluster(candidates)
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
