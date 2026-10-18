"""Сигнал «дверь кассы не в том виде, что в начале ролика» для R15 (без моделей, только numpy).

Дверь стеклянная: вопрос SigLIP «открыта / закрыта» по её вырезке ненадёжен (SceneConfig.enable_door_state
выключен). Вместо него — сравнение с видом двери в первые секунды ролика, как R10 сравнивает сейф с уровнем
первых секунд. Вырезка зоны двери (по высоте — полоса band: внизу проёма стоит стул клиента) сжимается до
сетки DOOR_GRID в оттенках серого и нормируется (среднее 0, разброс 1) — общий сдвиг яркости и контраста
не меняет картину. Клетки, которые накрывает любой человек (и за стеклом тоже), в сравнение не входят.

Замер 18.10 по кадрам (зона двери размечена вручную, люди не закрашены): дверь оставлена открытой —
Х-32 0,77–0,84 и Х-33 0,87–0,99 больше 10 с; вход с закрытием двери (Х-3, Х-5, Х-19, Х-31, Х-35, Х-37),
уход кассира (Х-15, Х-22–Х-29) — всплеск 0,5–1,0 на 2–4 с прохода, потом 0,03–0,2. Свет гаснет при уходе
(Х-11, Х-12) — 1,0 до конца ролика: это отсекает детектор по яркости кадра (worker.events.detectors).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

from worker.perception.geometry import BBox

__all__ = ["DOOR_GRID", "door_band", "door_patch", "door_diffs"]

DOOR_GRID = (24, 48)            # ширина и высота сетки вырезки, клеток
_MIN_CROP_PX = 6


def door_band(door: BBox, top: float, bottom: float) -> BBox:
    """Полоса зоны двери по высоте: от top до bottom доли её высоты (0 — верх двери)."""
    h = door[3] - door[1]
    return (door[0], door[1] + top * h, door[2], door[1] + bottom * h)


def _resize_gray(crop: Any, grid: tuple[int, int]) -> Any:
    """Вырезка HxWx3 (BGR) или HxW -> серая сетка grid[1] x grid[0] (среднее по клеткам), float32."""
    import numpy as np

    gray = crop.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gw, gh = grid
    try:
        import cv2

        return cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    except ImportError:     # без OpenCV — среднее по блокам
        rows = np.linspace(0, gray.shape[0], gh + 1).astype(int)
        cols = np.linspace(0, gray.shape[1], gw + 1).astype(int)
        out = np.empty((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                block = gray[rows[i]:max(rows[i + 1], rows[i] + 1), cols[j]:max(cols[j + 1], cols[j] + 1)]
                out[i, j] = block.mean()
        return out


def door_patch(image: Any, band: BBox, person_boxes: Sequence[BBox], *, margin: float = 0.02,
               grid: tuple[int, int] = DOOR_GRID) -> Optional[tuple[Any, Any]]:
    """Серая сетка полосы двери и маска клеток, которые накрывает человек (True — закрыто человеком).

    band и person_boxes — доли кадра (x1, y1, x2, y2); рамки людей расширяются на margin. None —
    вырезка меньше _MIN_CROP_PX пикселей по стороне.
    """
    import numpy as np

    h, w = image.shape[:2]
    x1, y1 = max(0, int(band[0] * w)), max(0, int(band[1] * h))
    x2, y2 = min(w, int(round(band[2] * w))), min(h, int(round(band[3] * h)))
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None
    gray = _resize_gray(image[y1:y2, x1:x2], grid)
    gw, gh = grid
    # центры клеток в долях кадра
    xs = (x1 + (np.arange(gw) + 0.5) * (x2 - x1) / gw) / w
    ys = (y1 + (np.arange(gh) + 0.5) * (y2 - y1) / gh) / h
    mask = np.zeros((gh, gw), dtype=bool)
    for b in person_boxes:
        inside_x = (xs >= b[0] - margin) & (xs <= b[2] + margin)
        inside_y = (ys >= b[1] - margin) & (ys <= b[3] + margin)
        mask |= inside_y[:, None] & inside_x[None, :]
    return gray, mask


def _normalized(gray: Any, mask: Any) -> Optional[Any]:
    """Сетка, нормированная по открытым клеткам (среднее 0, разброс 1); закрытые людьми — NaN."""
    import numpy as np

    free = gray[~mask]
    if free.size < 2:
        return None
    z = (gray - free.mean()) / (free.std() + 1e-6)
    z = z.astype(np.float32)
    z[mask] = np.nan
    return z


def door_diffs(patches: Sequence[Optional[tuple[Any, Any]]], times: Sequence[float], *,
               baseline_sec: float = 1.0, max_masked: float = 0.5) -> list[Optional[float]]:
    """Насколько полоса двери в каждом кадре отличается от её вида в первые baseline_sec секунд.

    Значение — среднее |z - z_эталон| по клеткам, открытым и в кадре, и в эталоне (у одинаковых картинок
    около 0, у независимых около 1,1). Эталон — медиана по клеткам кадров первых baseline_sec секунд, где
    люди закрывают не больше max_masked полосы. None — кадра нет, люди закрывают больше max_masked или
    эталона нет (в начале ролика дверь всё время закрыта людьми).
    """
    import numpy as np

    if len(patches) != len(times):
        raise ValueError("door_diffs: patches и times разной длины")
    if not patches:
        return []
    zs: list[Optional[Any]] = []
    for patch in patches:
        if patch is None:
            zs.append(None)
            continue
        gray, mask = patch
        zs.append(None if mask.mean() > max_masked else _normalized(gray, mask))
    t0 = times[0]
    base_stack = [z for z, t in zip(zs, times) if z is not None and t <= t0 + baseline_sec + 1e-6]
    if not base_stack:
        return [None] * len(patches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # клетка закрыта во всех кадрах эталона
        base = np.nanmedian(np.stack(base_stack), axis=0)
    out: list[Optional[float]] = []
    min_free = 1.0 - max_masked
    for z in zs:
        if z is None:
            out.append(None)
            continue
        both = ~np.isnan(z) & ~np.isnan(base)
        if both.mean() < min_free:
            out.append(None)
            continue
        out.append(round(float(np.abs(z[both] - base[both]).mean()), 4))
    return out
