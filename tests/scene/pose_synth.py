"""Синтетические позы для тестов: сидящий кассир у стола, кадр 464×848 (как в датасете).

Точки задаются в пикселях вокруг центра плеч; S — ширина плеч, пиксели. factor уменьшает всю
сцену (позу, рамку, зоны) вокруг ORIGIN — для проверки инвариантности признаков к масштабу.
"""

from __future__ import annotations

from typing import Mapping, Optional

from worker.perception.pose import NUM_KEYPOINTS, Keypoints

FRAME = (464, 848)
ORIGIN = (232.0, 400.0)          # середина плеч кассира
S = 60.0                         # ширина плеч, px

# Сидящий кассир лицом к камере (px).
SEATED: dict[int, tuple[float, float]] = {
    0: (232, 360), 1: (222, 352), 2: (242, 352), 3: (212, 358), 4: (252, 358),
    5: (262, 400), 6: (202, 400), 7: (272, 450), 8: (192, 450), 9: (250, 490), 10: (214, 490),
    11: (252, 480), 12: (212, 480), 13: (255, 505), 14: (209, 505), 15: (255, 560), 16: (209, 560),
}
STANDING_LEGS: dict[int, tuple[float, float]] = {13: (255, 555), 14: (209, 555), 15: (255, 630), 16: (209, 630)}

DESK_PX = (150.0, 460.0, 320.0, 530.0)
SAFE_PX = (380.0, 300.0, 450.0, 420.0)
CASHIER_BOX_PX = (180.0, 330.0, 290.0, 580.0)


def _scale_pt(p: tuple[float, float], factor: float) -> tuple[float, float]:
    return (ORIGIN[0] + (p[0] - ORIGIN[0]) * factor, ORIGIN[1] + (p[1] - ORIGIN[1]) * factor)


def box(px: tuple[float, float, float, float], factor: float = 1.0) -> tuple[float, float, float, float]:
    """Рамка в px -> доли кадра (с масштабом вокруг ORIGIN)."""
    x1, y1 = _scale_pt((px[0], px[1]), factor)
    x2, y2 = _scale_pt((px[2], px[3]), factor)
    return (x1 / FRAME[0], y1 / FRAME[1], x2 / FRAME[0], y2 / FRAME[1])


def zones(factor: float = 1.0) -> dict[str, Optional[tuple[float, float, float, float]]]:
    return {"desk": box(DESK_PX, factor), "safe": box(SAFE_PX, factor), "door": None, "workplace": None, "laptop": None}


def kp(
    overrides: Optional[Mapping[int, tuple[float, float]]] = None,
    *,
    factor: float = 1.0,
    score: float = 0.9,
    hidden: tuple[int, ...] = (),
    base: Optional[Mapping[int, tuple[float, float]]] = None,
) -> Keypoints:
    """Keypoints из px-точек позы base (по умолчанию SEATED) с заменами overrides; hidden — score 0,05."""
    pts = dict(base or SEATED)
    pts.update(overrides or {})
    out = []
    for i in range(NUM_KEYPOINTS):
        x, y = _scale_pt(pts[i], factor)
        out.append((x / FRAME[0], y / FRAME[1], 0.05 if i in hidden else score))
    return Keypoints(tuple(out))


def wrist_at(dx: float, dy: float) -> tuple[float, float]:
    """Точка в px относительно ORIGIN, в единицах S."""
    return (ORIGIN[0] + dx * S, ORIGIN[1] + dy * S)
