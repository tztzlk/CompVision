"""Головы жестов кассира поверх точек позы — обучаемая часть гибрида (признаки — stdlib, инференс — numpy).

Учим жесты, а не нарушения: на тип нарушения в датасете 1–2 примера, а жесты встречаются в десятках
видео, в том числе в чистых. Нарушение = жест + контекст (детекторы worker.events.pose_detectors).

  жест           поле геометрии (pose_features)    «да»
  hand_in_safe   cashier_hand_in_safe              кисть кассира на сейфе или в нём
  handover       hand_to_hand                      кисть кассира встречает кисть другого человека (передача)
  counting       cashier_counting                  работа с купюрами на столе: руками или у счётной машинки
  hand_at_face   cashier_hand_at_face              кисть у лица (напиток, еда, телефон у уха)
  standing       cashier_standing                  кассир стоит или идёт

Вход головы — window_features: признаки кассира в окне ±1 с вокруг состояния. Координаты точек — от
середины плеч в единицах s (ширина плеч, worker.perception.pose.scale); расстояния кистей — до зон
стола и сейфа, до головы, до кистей и рамок других людей в комнате; скорость кистей; высота рамки к
её медиане по видео; сами признаки геометрии. Сырые координаты кадра не подаются (они запоминают
комнату), лево и право симметризованы (|dx|, min/max по рукам): отражение кадра признаки не меняет.

Веса — models/heads/pose_gestures.npz (scripts/train_pose_heads.py): на жест — стандартизация, слои
(логистическая регрессия или MLP с одним скрытым слоем, ReLU) и режим гибрида, выбранный по замеру
на фолдах по видео:
  geometry — голова не используется (на замере не лучше геометрии);
  band     — вероятность головы заменяет геометрию только в спорной полосе band_lo < g < band_hi;
  fill     — как band, и ещё там, где геометрия None, а точки кассира есть;
  head     — вероятность головы везде, где она есть.
Вероятность головы пишется и отдельным полем <поле геометрии>_head (правило 7 контракта).
Порог решения головы при сохранении вшит в смещение: вероятность 0,5 = порог, как у геометрии.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional, Sequence

from worker.perception.pose import (
    LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE,
    RIGHT_SHOULDER, RIGHT_WRIST, Keypoints, dist_px, head_point, point, scale, wrists,
)
from worker.scene.pose_features import FEATURE_NAMES as GEOMETRY_NAMES
from worker.scene.pose_features import PoseFrame

__all__ = [
    "GESTURES",
    "GEOMETRY_OF",
    "MODES",
    "FEATURE_NAMES",
    "GestureHead",
    "state_features",
    "window_features",
    "hybrid_value",
    "load_gesture_heads",
    "apply_gesture_heads",
]

LOG = logging.getLogger("worker.scene.pose_heads")

GESTURES: tuple[str, ...] = ("hand_in_safe", "handover", "counting", "hand_at_face", "standing")
GEOMETRY_OF: Mapping[str, str] = {
    "hand_in_safe": "cashier_hand_in_safe",
    "handover": "hand_to_hand",
    "counting": "cashier_counting",
    "hand_at_face": "cashier_hand_at_face",
    "standing": "cashier_standing",
}
MODES: tuple[str, ...] = ("geometry", "band", "fill", "head")

WINDOW_SEC = 1.0             # окно признаков: ±1 с вокруг состояния
CAP = 6.0                    # расстояния и координаты в s ограничены: дальше — просто «далеко»
MIN_SCORE = 0.3              # как DEFAULT_MIN_SCORE позы

Point = tuple[float, float]
BBox = tuple[float, float, float, float]

_ARMS = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))
# Признаки одной руки; для расстояний «нет данных» = CAP (далеко), для координат — 0.
_ARM_DIST = ("desk", "safe", "head", "other", "other_box")
_ARM_POS = ("adx", "dy", "reach", "elbow_dy")
_ARM_KEYS = _ARM_POS + _ARM_DIST
_BASE_KEYS: tuple[str, ...] = (
    "vis_wrists", "vis_elbows", "vis_hips", "vis_knees", "vis_head",
    "head_dy", "hip_dy", "thigh_ratio", "thigh_ok", "wrists_gap",
    "box_ok", "box_aspect", "box_rel_height", "box_h_s", "body_desk", "body_safe", "n_others",
) + tuple(f"arm_{k}_{agg}" for k in _ARM_KEYS for agg in ("min", "max"))
# Ряды, по которым в окне берутся min и max — только по состояниям, где величина наблюдалась
# (флаг-ключ); не наблюдалась нигде в окне — заглушка текущего состояния.
_WIN_SOURCE: tuple[tuple[str, str], ...] = (
    ("arm_desk_min", "vis_wrists"), ("arm_safe_min", "vis_wrists"), ("arm_head_min", "vis_wrists"),
    ("arm_other_min", "vis_wrists"), ("arm_dy_min", "vis_wrists"),
    ("box_rel_height", "box_ok"), ("box_h_s", "box_ok"), ("thigh_ratio", "thigh_ok"),
)
_WIN_KEYS: tuple[str, ...] = tuple(f"win_{k}_{agg}" for k, _ in _WIN_SOURCE for agg in ("min", "max")) + (
    "win_vis_wrists_mean", "win_thigh_ok_share", "win_wrist_speed", "win_height_rate")
_GEO_KEYS: tuple[str, ...] = tuple(f"geo_{n}" for n in GEOMETRY_NAMES) + tuple(f"geo_{n}_ok" for n in GEOMETRY_NAMES)
FEATURE_NAMES: tuple[str, ...] = _BASE_KEYS + _WIN_KEYS + _GEO_KEYS


# --------------------------------------------------------------------------- признаки (stdlib)


def _cap(v: float) -> float:
    return max(-CAP, min(CAP, v))


def _box_dist_px(p: Point, box: BBox, frame_size: tuple[int, int]) -> float:
    """Расстояние от точки до рамки в пикселях: снаружи — до ближайшего края, внутри — со знаком минус."""
    w, h = frame_size
    x, y = p[0] * w, p[1] * h
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    if dx == 0.0 and dy == 0.0:
        return -min(x - x0, x1 - x, y - y0, y1 - y)
    return math.hypot(dx, dy)


def _mean(points: Sequence[Point]) -> Optional[Point]:
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _origin(kp: Keypoints) -> Optional[Point]:
    """Опорная точка — середина плеч; одно плечо — оно; плеч нет — голова."""
    shoulders = [p for p in (point(kp, LEFT_SHOULDER, MIN_SCORE), point(kp, RIGHT_SHOULDER, MIN_SCORE)) if p]
    return _mean(shoulders) or head_point(kp, MIN_SCORE)


def _thigh_ratio(kp: Keypoints, frame_size: tuple[int, int]) -> Optional[float]:
    """Вертикаль бедра к вертикали торса (как в pose_features._standing_by_legs), максимум по сторонам."""
    h = frame_size[1]
    ratios: list[float] = []
    for sh_i, hip_i, knee_i in ((LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)):
        sh, hip, knee = point(kp, sh_i, MIN_SCORE), point(kp, hip_i, MIN_SCORE), point(kp, knee_i, MIN_SCORE)
        if sh is None or hip is None or knee is None:
            continue
        torso = (hip[1] - sh[1]) * h
        if torso > 0:
            ratios.append((knee[1] - hip[1]) * h / torso)
    return max(-2.0, min(3.0, max(ratios))) if ratios else None


def state_features(
    frame: PoseFrame,
    zones: Mapping[str, Optional[BBox]],
    frame_size: tuple[int, int],
    seated_height: Optional[float] = None,
) -> Optional[dict[str, float]]:
    """Признаки кассира в одном состоянии (ключи _BASE_KEYS) или None, если точек или масштаба нет.

    seated_height — медиана высоты рамки кассира по видео (доля кадра): «рамка выше обычного» = встал.
    """
    kp = frame.cashier_kp
    if kp is None:
        return None
    s = scale(kp, frame_size, frame.cashier_bbox, MIN_SCORE)
    o = _origin(kp)
    if s is None or s <= 0 or o is None:
        return None
    w, h = frame_size

    def rel(p: Point) -> Point:
        return ((p[0] - o[0]) * w / s, (p[1] - o[1]) * h / s)

    desk, safe = zones.get("desk"), zones.get("safe")
    head = head_point(kp, MIN_SCORE)
    other_wrists = [p for person in frame.others for p in wrists(person.kp, MIN_SCORE)]
    other_boxes = [person.bbox for person in frame.others]

    arms: list[dict[str, float]] = []
    wpts: list[Point] = []
    by_index: dict[int, Point] = {}
    for sh_i, el_i, wr_i in _ARMS:
        wr = point(kp, wr_i, MIN_SCORE)
        if wr is None:
            continue
        wpts.append(wr)
        by_index[wr_i] = wr
        sh = point(kp, sh_i, MIN_SCORE) or o
        el = point(kp, el_i, MIN_SCORE)
        rx, ry = rel(wr)
        arms.append({
            "adx": _cap(abs(rx)),
            "dy": _cap(ry),
            "reach": _cap(dist_px(wr, sh, frame_size) / s),
            "elbow_dy": _cap(rel(el)[1]) if el is not None else _cap(ry / 2),
            "desk": _cap(_box_dist_px(wr, desk, frame_size) / s) if desk is not None else CAP,
            "safe": _cap(_box_dist_px(wr, safe, frame_size) / s) if safe is not None else CAP,
            "head": _cap(dist_px(wr, head, frame_size) / s) if head is not None else CAP,
            "other": _cap(min(dist_px(wr, q, frame_size) for q in other_wrists) / s) if other_wrists else CAP,
            "other_box": _cap(min(_box_dist_px(wr, b, frame_size) for b in other_boxes) / s) if other_boxes else CAP,
        })

    f: dict[str, float] = {
        "vis_wrists": len(arms) / 2,
        "vis_elbows": sum(1 for i in (LEFT_ELBOW, RIGHT_ELBOW) if point(kp, i, MIN_SCORE)) / 2,
        "vis_hips": sum(1 for i in (LEFT_HIP, RIGHT_HIP) if point(kp, i, MIN_SCORE)) / 2,
        "vis_knees": sum(1 for i in (LEFT_KNEE, RIGHT_KNEE) if point(kp, i, MIN_SCORE)) / 2,
        "vis_head": 1.0 if head is not None else 0.0,
        "head_dy": _cap(rel(head)[1]) if head is not None else 0.0,
        "wrists_gap": _cap(dist_px(wpts[0], wpts[1], frame_size) / s) if len(wpts) == 2 else CAP,
        "body_desk": _cap(_box_dist_px(o, desk, frame_size) / s) if desk is not None else CAP,
        "body_safe": _cap(_box_dist_px(o, safe, frame_size) / s) if safe is not None else CAP,
        "n_others": float(min(len(frame.others), 3)),
    }
    hip_mid = _mean([p for p in (point(kp, LEFT_HIP, MIN_SCORE), point(kp, RIGHT_HIP, MIN_SCORE)) if p])
    f["hip_dy"] = _cap(rel(hip_mid)[1]) if hip_mid is not None else 0.0
    thigh = _thigh_ratio(kp, frame_size)
    f["thigh_ratio"] = thigh if thigh is not None else 0.0
    f["thigh_ok"] = 1.0 if thigh is not None else 0.0
    box = frame.cashier_bbox
    f["box_ok"] = 1.0 if box is not None else 0.0
    if box is not None:
        bh, bw = (box[3] - box[1]) * h, (box[2] - box[0]) * w
        f["box_aspect"] = _cap(bh / bw) if bw > 0 else 0.0
        f["box_rel_height"] = _cap((box[3] - box[1]) / seated_height) if seated_height else 1.0
        f["box_h_s"] = _cap(bh / s)
    else:
        f["box_aspect"], f["box_rel_height"], f["box_h_s"] = 0.0, 1.0, 0.0
    for key in _ARM_KEYS:
        vals = [a[key] for a in arms]
        default = CAP if key in _ARM_DIST else 0.0
        f[f"arm_{key}_min"] = min(vals) if vals else default
        f[f"arm_{key}_max"] = max(vals) if vals else default
    f["_s"] = s
    f["_wrists"] = by_index  # type: ignore[assignment]  # служебное: кисти по индексу точки — для скорости в окне
    return f


def _wrist_speed(base: Sequence[Optional[dict[str, Any]]], times: Sequence[float], idx: Sequence[int],
                 frame_size: tuple[int, int]) -> float:
    """Средняя скорость кистей в окне, s в секунду: левая кисть — с левой, правая — с правой, только если
    кисть видна в обоих соседних состояниях (появление второй кисти — не движение)."""
    total, steps = 0.0, 0
    for a, b in zip(idx, idx[1:]):
        pa, pb = base[a]["_wrists"], base[b]["_wrists"]  # type: ignore[index]
        dt = times[b] - times[a]
        if dt <= 0:
            continue
        for k in set(pa) & set(pb):
            total += dist_px(pa[k], pb[k], frame_size) / base[b]["_s"] / dt  # type: ignore[index]
            steps += 1
    return _cap(total / steps) if steps else 0.0


def window_features(
    frames: Sequence[PoseFrame],
    geometry: Sequence[Mapping[str, Any]],
    zones: Mapping[str, Optional[BBox]],
    frame_size: tuple[int, int],
) -> list[Optional[list[float]]]:
    """Вектор признаков (порядок FEATURE_NAMES) на каждое состояние; None — точек кассира нет.

    geometry — признаки геометрии по тем же состояниям (compute_features или поля сцены).
    """
    heights = [f.cashier_bbox[3] - f.cashier_bbox[1] for f in frames if f.cashier_bbox is not None]
    seated = median(heights) if heights else None
    base = [state_features(f, zones, frame_size, seated) for f in frames]
    times = [f.t for f in frames]
    steps = [b - a for a, b in zip(times, times[1:]) if b > a]
    # Допуск — полшага выборки: окно ±1 с одинаково при 30 и 29,97 к/с исходника.
    tol = WINDOW_SEC + (median(steps) / 2 if steps else 0.0)
    out: list[Optional[list[float]]] = []
    for i, b in enumerate(base):
        if b is None:
            out.append(None)
            continue
        idx = [j for j in range(len(frames)) if base[j] is not None and abs(times[j] - times[i]) < tol]
        row: dict[str, float] = {k: b[k] for k in _BASE_KEYS}
        for key, flag in _WIN_SOURCE:
            vals = [base[j][key] for j in idx if base[j][flag] > 0] or [b[key]]  # type: ignore[index]
            row[f"win_{key}_min"] = min(vals)
            row[f"win_{key}_max"] = max(vals)
        row["win_vis_wrists_mean"] = sum(base[j]["vis_wrists"] for j in idx) / len(idx)  # type: ignore[index]
        row["win_thigh_ok_share"] = sum(base[j]["thigh_ok"] for j in idx) / len(idx)  # type: ignore[index]
        row["win_wrist_speed"] = _wrist_speed(base, times, idx, frame_size)
        boxed = [j for j in idx if base[j]["box_ok"] > 0]  # type: ignore[index]
        span = times[boxed[-1]] - times[boxed[0]] if len(boxed) > 1 else 0.0
        row["win_height_rate"] = (_cap((base[boxed[-1]]["box_rel_height"] - base[boxed[0]]["box_rel_height"]) / span)  # type: ignore[index]
                                  if span > 0 else 0.0)
        geo = geometry[i] if i < len(geometry) else {}
        for name in GEOMETRY_NAMES:
            v = geo.get(name)
            row[f"geo_{name}"] = float(v) if v is not None else 0.0
            row[f"geo_{name}_ok"] = 1.0 if v is not None else 0.0
        out.append([float(row[k]) for k in FEATURE_NAMES])
    return out


# --------------------------------------------------------------------------- головы (numpy)


@dataclass(frozen=True)
class GestureHead:
    """Голова одного жеста: стандартизация, слои [(W, b)] (1 слой — логрегрессия, 2 — MLP с ReLU), режим гибрида."""

    gesture: str
    mean: Any
    std: Any
    layers: tuple[tuple[Any, Any], ...]
    mode: str = "band"
    band_lo: float = 0.2
    band_hi: float = 0.8

    def predict(self, X: Any) -> Any:
        """Вероятность жеста на каждую строку X (n × len(FEATURE_NAMES))."""
        import numpy as np

        z = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            z = z @ W + b
            if k < len(self.layers) - 1:
                z = np.maximum(z, 0.0)
        z = np.clip(z.reshape(-1), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))


def hybrid_value(geo: Optional[float], prob: Optional[float], head: GestureHead) -> Optional[float]:
    """Значение поля после гибрида: где решает геометрия, а где голова (см. режимы в docstring модуля)."""
    if prob is None or head.mode == "geometry":
        return geo
    if head.mode == "head":
        return prob
    if geo is None:
        return prob if head.mode == "fill" else None
    if head.band_lo < geo < head.band_hi:
        return prob
    return geo


def load_gesture_heads(path: Path) -> dict[str, GestureHead]:
    """Головы из .npz scripts/train_pose_heads.py; жесты с режимом geometry пропускаются.

    Список признаков в файле должен совпадать с FEATURE_NAMES — иначе ValueError (головы обучены
    на другой версии признаков и молча давали бы мусор).
    """
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as data:
        names = tuple(str(x) for x in data["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError(f"{path}: признаки головы ({len(names)}) не совпадают с FEATURE_NAMES "
                             f"({len(FEATURE_NAMES)}) — переобучите scripts/train_pose_heads.py")
        meta = json.loads(str(data["meta"]))
        heads: dict[str, GestureHead] = {}
        for gesture, info in meta.get("heads", {}).items():
            if gesture not in GEOMETRY_OF:
                raise ValueError(f"{path}: неизвестный жест {gesture!r}")
            mode = info.get("mode", "geometry")
            if mode not in MODES:
                raise ValueError(f"{path}: неизвестный режим {mode!r} у {gesture}")
            if mode == "geometry":
                continue
            layers = tuple((np.array(data[f"{gesture}__W{k}"], dtype=np.float64),
                            np.array(data[f"{gesture}__b{k}"], dtype=np.float64))
                           for k in range(int(info["layers"])))
            heads[gesture] = GestureHead(
                gesture=gesture,
                mean=np.array(data[f"{gesture}__mean"], dtype=np.float64),
                std=np.array(data[f"{gesture}__std"], dtype=np.float64),
                layers=layers,
                mode=mode,
                band_lo=float(info.get("band_lo", 0.2)),
                band_hi=float(info.get("band_hi", 0.8)),
            )
    return heads


def apply_gesture_heads(
    frames: Sequence[PoseFrame],
    features: Sequence[Mapping[str, Any]],
    zones: Mapping[str, Optional[BBox]],
    frame_size: tuple[int, int],
    heads: Mapping[str, GestureHead],
) -> list[dict[str, Any]]:
    """Признаки после гибрида: поле геометрии по режиму головы и поле <поле>_head с вероятностью головы."""
    rows = [dict(r) for r in features]
    if not heads:
        return rows
    X = window_features(frames, features, zones, frame_size)
    idx = [i for i, x in enumerate(X) if x is not None]
    if not idx:
        return rows
    import numpy as np

    M = np.asarray([X[i] for i in idx], dtype=np.float64)
    for gesture, head in heads.items():
        field = GEOMETRY_OF[gesture]
        probs = head.predict(M)
        for i, p in zip(idx, probs):
            prob = round(float(p), 3)
            rows[i][f"{field}_head"] = prob
            rows[i][field] = hybrid_value(rows[i].get(field), prob, head)
    return rows
