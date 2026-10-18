"""Поза кассира для ухода с рабочего места (SceneConfig.pose_departure, по умолчанию выключено).

Точки ViTPose (17 точек COCO, [[x, y, score], ...] в долях кадра) приходят в builder из сырого кэша
(поле person_pose, scripts/add_pose_to_raw.py) — в проде позы нет (нужен transformers >= 4.48).
Три признака, каждый — отдельная функция и подфлаг SceneConfig:

  feet_behind_glass — ноги видны (колено и лодыжка с уверенностью >= pose_leg_score), а ступни не ниже
      верхнего края стола больше чем на pose_foot_band: человек целиком виден за стеклом в коридоре.
      Замер 18.10 (data/scene_cache_r3_pose): у сидящего кассира лодыжки закрыты столом (0,02–0,08),
      у кассира за стеклом до входа (Х-31 2,7–6,7 с, Х-32 2,0–3,3 с) колени 0,7–0,8, лодыжки 0,55–0,69,
      ступни на 0–0,02 кадра ниже верха стола (на flip лодыжки 0,44–0,49 — порог 0,4, не 0,5).
      Такая точка не «у места» (pose_dep_glass).
  anchor_point — середина таза вместо центра рамки для расстояния до места (pose_dep_anchor):
      рамка стоящего у края стола растягивается в сторону, таз — нет.
  is_standing — бедро от таза к колену почти вертикально (pose_stand_ratio): момент «встал» для
      таймкода ухода вместо роста рамки leave_up_ratio (pose_dep_stand).
Нет точек на кадре (ViTPose не запускался, низкая уверенность) — поведение как без позы.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

__all__ = ["KNEES", "ANKLES", "HIPS", "legs_visible", "feet_y", "feet_behind_glass", "anchor_point",
           "is_standing", "pose_by_frame", "match_pose", "pose_departure_inputs", "stood_up_at"]

HIPS = (11, 12)
KNEES = (13, 14)
ANKLES = (15, 16)

Kp = Sequence[Sequence[float]]


def _visible(kp: Kp, idx: Sequence[int], min_score: float) -> list[Sequence[float]]:
    return [kp[i] for i in idx if i < len(kp) and kp[i][2] >= min_score]


def legs_visible(kp: Optional[Kp], min_score: float) -> bool:
    """Видны ли ноги: хотя бы одно колено и одна лодыжка с уверенностью >= min_score."""
    if not kp:
        return False
    return bool(_visible(kp, KNEES, min_score)) and bool(_visible(kp, ANKLES, min_score))


def feet_y(kp: Optional[Kp], min_score: float) -> Optional[float]:
    """Высота ступней (средняя y видимых лодыжек) или None."""
    if not kp:
        return None
    ankles = _visible(kp, ANKLES, min_score)
    return sum(a[1] for a in ankles) / len(ankles) if ankles else None


def feet_behind_glass(kp: Optional[Kp], desk_top: Optional[float], min_score: float, band: float) -> bool:
    """Человек виден во весь рост, а ступни на уровне верха стола или выше — он за стеклом, не у места.

    Сидящий за столом кассир — ноги закрыты столом (False); стоящий в комнате перед столом — ступни
    заметно ниже верха стола (False).
    """
    if desk_top is None or not legs_visible(kp, min_score):
        return False
    fy = feet_y(kp, min_score)
    return fy is not None and fy <= desk_top + band


def anchor_point(kp: Optional[Kp], min_score: float) -> Optional[tuple[float, float]]:
    """Середина таза (оба бедра с уверенностью >= min_score) или None."""
    if not kp:
        return None
    hips = _visible(kp, HIPS, min_score)
    if len(hips) < 2:
        return None
    return (hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2


def is_standing(kp: Optional[Kp], min_score: float, ratio: float) -> Optional[bool]:
    """Стоит ли человек: бедро (таз -> колено) ближе к вертикали, чем ratio (|dy| / длина бедра).

    Сидя бедро почти горизонтально (|dy| / длина мало), стоя — вертикально. None — нет пары таз+колено.
    """
    if not kp:
        return None
    best: Optional[float] = None
    for hip, knee in zip(HIPS, KNEES):
        if kp[hip][2] < min_score or kp[knee][2] < min_score:
            continue
        dx, dy = kp[knee][0] - kp[hip][0], kp[knee][1] - kp[hip][1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        v = dy / length
        best = v if best is None else max(best, v)
    return None if best is None else best >= ratio


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pose_by_frame(person_pose: Optional[Sequence[Sequence[Any]]]) -> dict[int, list[tuple[Sequence[float], Kp]]]:
    """[(позиция кадра, рамка, точки)] из сырого кэша -> {позиция: [(рамка, точки)]}."""
    out: dict[int, list[tuple[Sequence[float], Kp]]] = {}
    for fi, box, kp in person_pose or []:
        out.setdefault(int(fi), []).append((box, kp))
    return out


def match_pose(index: dict[int, list[tuple[Sequence[float], Kp]]], pos: int,
               box: Optional[Sequence[float]], min_iou: float = 0.9) -> Optional[Kp]:
    """Точки той рамки на кадре pos, что совпадает с box (IoU >= min_iou), иначе None."""
    if box is None:
        return None
    best, best_iou = None, min_iou
    for b, kp in index.get(pos, ()):
        v = _iou(b, box)
        if v >= best_iou:
            best, best_iou = kp, v
    return best


def pose_departure_inputs(boxes: Sequence[Optional[Sequence[float]]], pose: Sequence[Optional[Kp]],
                          workplace: Sequence[float], desk: Optional[Sequence[float]], config: Any
                          ) -> tuple[list[float], list[bool], Optional[list[Optional[bool]]]]:
    """Входы builder._departures с позой: расстояние до места (в ширинах места), «не у места» по позе,
    «стоит» по кадрам (None — подфлаг pose_dep_stand выключен)."""
    wx, wy = (workplace[0] + workplace[2]) / 2, (workplace[1] + workplace[3]) / 2
    width = max(workplace[2] - workplace[0], 1e-6)
    desk_top = desk[1] if desk is not None else None
    dist: list[float] = []
    not_home: list[bool] = []
    for box, kp in zip(boxes, pose):
        if box is None:
            dist.append(0.0)
            not_home.append(False)
            continue
        point = anchor_point(kp, config.pose_leg_score) if config.pose_dep_anchor else None
        cx, cy = point if point is not None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        dist.append(math.hypot(cx - wx, cy - wy) / width)
        not_home.append(bool(config.pose_dep_glass)
                        and feet_behind_glass(kp, desk_top, config.pose_leg_score, config.pose_foot_band))
    standing = ([is_standing(kp, config.pose_leg_score, config.pose_stand_ratio) for kp in pose]
                if config.pose_dep_stand else None)
    return dist, not_home, standing


def stood_up_at(standing: Sequence[Optional[bool]], times: Sequence[float], last_home: int, up_sec: float) -> int:
    """Таймкод ухода по позе: первый кадр подряд «стоит» перед last_home, если перед ним кассир сидел
    не раньше чем за up_sec; иначе last_home (как builder._departures по росту рамки)."""
    if not standing[last_home]:
        return last_home
    k = last_home
    while k > 0 and times[last_home] - times[k - 1] <= up_sec and standing[k - 1]:
        k -= 1
    if k > 0 and standing[k - 1] is False and times[last_home] - times[k - 1] <= up_sec:
        return k
    return last_home
