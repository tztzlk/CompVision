"""Геометрия рамок в долях кадра: 0..1, [x1, y1, x2, y2]. Только stdlib."""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable, Optional, Sequence

__all__ = [
    "BBox",
    "clip",
    "width",
    "height",
    "area",
    "center",
    "expand",
    "pad",
    "iou",
    "ioa",
    "contains_point",
    "center_dist",
    "median_box",
    "nms",
]

BBox = tuple[float, float, float, float]


def clip(box: Sequence[float]) -> BBox:
    """Обрезать рамку по границам кадра и упорядочить координаты."""
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def area(box: BBox) -> float:
    return width(box) * height(box)


def center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def expand(box: BBox, frac: float) -> BBox:
    """Расширить рамку на frac её ширины/высоты с каждой стороны (с обрезкой по кадру)."""
    dx, dy = width(box) * frac, height(box) * frac
    return clip((box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy))


def pad(box: BBox, dx: float, dy: Optional[float] = None) -> BBox:
    """Расширить рамку на абсолютные dx/dy (доли кадра) с каждой стороны."""
    dy = dx if dy is None else dy
    return clip((box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy))


def _intersection(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union."""
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(a: BBox, b: BBox) -> float:
    """Доля площади a внутри b (intersection over area of a)."""
    a_area = area(a)
    return _intersection(a, b) / a_area if a_area > 0 else 0.0


def contains_point(box: BBox, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def center_dist(a: BBox, b: BBox) -> float:
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(ax - bx, ay - by)


def median_box(boxes: Iterable[BBox]) -> Optional[BBox]:
    """Покоординатная медиана рамок (None для пустого списка)."""
    items = list(boxes)
    if not items:
        return None
    return clip(tuple(median(b[i] for b in items) for i in range(4)))


def nms(boxes: Sequence[tuple[BBox, float]], iou_threshold: float) -> list[int]:
    """Индексы рамок, оставшихся после подавления дублей (по убыванию score)."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1], reverse=True)
    kept: list[int] = []
    for i in order:
        if all(iou(boxes[i][0], boxes[j][0]) < iou_threshold for j in kept):
            kept.append(i)
    return sorted(kept)
