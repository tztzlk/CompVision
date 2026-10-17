"""Признаки позы по ключевым точкам: что делают руки кассира (только stdlib).

Каждый признак на каждое состояние — число 0..1 или None. None = нет данных (точка не видна,
зона не найдена, кассир не определён) и НЕ равен «нарушения нет» (правило 1 контракта).
Пороги — в единицах масштаба человека s (worker.perception.pose.scale: ширина плеч в
пикселях), а не в пикселях: финальное видео снято под другим углом и в другом размере.

Признаки (поля состояния, правило 7 контракта):
  cashier_hand_in_safe   — видимое запястье кассира в зоне сейфа, расширенной на 0,5·s;
  cashier_hands_on_desk  — доля видимых запястий кассира в зоне стола (0, 0,5, 1);
  hand_to_hand           — запястье кассира ближе 0,5·s к запястью другого человека в комнате,
                           середина между ними над столом; hand_to_hand_track — его track_id;
  cashier_counting       — обе кисти над столом ближе 0,8·s друг к другу, кисти двигаются
                           (путь за окно 2 с выше порога), нет hand_to_hand;
  cashier_hand_at_face   — запястье ближе 0,6·s к голове (напиток, телефон у уха);
  cashier_standing       — бёдра заметно выше коленей (вертикаль бедра против торса); если ноги
                           закрыты столом — рамка кассира выше медианы своей высоты в 1,3 раза.

«Мягкие» значения: расстояние d при пороге thr даёт clamp(1,5 − d/thr): d ≤ 0,5·thr -> 1,
d = thr -> 0,5, d ≥ 1,5·thr -> 0. Так граница порога совпадает с 0,5, а confidence события
(среднее значения признака) показывает, насколько уверенно порог пройден.
Сглаживание — медиана по 3 точкам (как в остальном контракте), None в окне пропускаются.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Optional, Sequence

from worker.perception.pose import (
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
    Keypoints, dist_px, head_point, point, scale, wrists,
)

__all__ = [
    "FEATURE_NAMES",
    "PoseConfig",
    "PersonPose",
    "PoseFrame",
    "frame_features",
    "compute_features",
    "smooth3",
]

FEATURE_NAMES: tuple[str, ...] = (
    "cashier_hand_in_safe",
    "cashier_hands_on_desk",
    "hand_to_hand",
    "cashier_counting",
    "cashier_hand_at_face",
    "cashier_standing",
)

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class PoseConfig:
    """Пороги признаков позы; расстояния — в единицах s (ширина плеч человека)."""

    min_score: float = 0.3                  # точка с меньшим score не видна
    safe_margin_s: float = 0.5              # расширение зоны сейфа
    desk_margin_s: float = 0.25             # расширение зоны стола (кисть у края стола)
    hand_to_hand_s: float = 0.5             # запястья кассира и другого человека ближе
    counting_hands_s: float = 0.8           # кисти кассира ближе друг к другу
    counting_window_sec: float = 2.0        # окно движения кистей (по центру состояния)
    counting_motion_s: float = 1.0          # путь обеих кистей за окно, s (граница 0,5 значения)
    counting_min_points: int = 3            # состояний с видимыми кистями в окне не меньше
    face_s: float = 0.6                     # запястье ближе к голове
    standing_thigh_lo: float = 0.35         # вертикаль бедра / вертикаль торса: сидя (значение 0)
    standing_thigh_hi: float = 0.75         # стоя (значение 1)
    standing_bbox_lo: float = 1.15          # высота рамки / медиана высоты: значение 0
    standing_bbox_hi: float = 1.4           # значение 1 (1,3 -> 0,6)
    every_k: int = 1                        # поза на каждом k-м состоянии (если не укладываемся во время)
    smooth_points: int = 3


@dataclass(frozen=True)
class PersonPose:
    """Другой человек в комнате (кроме кассира): трек, рамка, точки."""

    track_id: int
    bbox: BBox
    kp: Optional[Keypoints]


@dataclass(frozen=True)
class PoseFrame:
    """Вход признаков на одно состояние: время, кассир и остальные люди с точками."""

    t: float
    cashier_kp: Optional[Keypoints] = None
    cashier_bbox: Optional[BBox] = None
    others: tuple[PersonPose, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- вспомогательное


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _closeness(d: float, thr: float) -> float:
    """d ≤ 0,5·thr -> 1; d = thr -> 0,5; d ≥ 1,5·thr -> 0."""
    return _clamp01(1.5 - d / thr) if thr > 0 else 0.0


def _ramp(x: float, lo: float, hi: float) -> float:
    return _clamp01((x - lo) / (hi - lo)) if hi > lo else (1.0 if x >= hi else 0.0)


def _inside(p: Point, box: BBox, margin_px: float, frame_size: tuple[int, int]) -> bool:
    """Точка (доли кадра) внутри рамки, расширенной на margin_px пикселей."""
    w, h = frame_size
    x, y = p[0] * w, p[1] * h
    return (box[0] * w - margin_px <= x <= box[2] * w + margin_px
            and box[1] * h - margin_px <= y <= box[3] * h + margin_px)


def smooth3(values: Sequence[Optional[float]], points: int = 3) -> list[Optional[float]]:
    """Скользящая медиана по центру окна; None в окне пропускаются, None остаётся None."""
    half = max(1, int(points)) // 2
    out: list[Optional[float]] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        window = [x for x in values[max(0, i - half): i + half + 1] if x is not None]
        out.append(round(median(window), 3))
    return out


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 3)


# --------------------------------------------------------------------------- признаки одного кадра


def _hand_in_zone(ws: Sequence[Point], zone: Optional[BBox], margin_px: float,
                  frame_size: tuple[int, int]) -> Optional[float]:
    if zone is None or not ws:
        return None
    return 1.0 if any(_inside(w, zone, margin_px, frame_size) for w in ws) else 0.0


def _hands_share(ws: Sequence[Point], zone: Optional[BBox], margin_px: float,
                 frame_size: tuple[int, int]) -> Optional[float]:
    if zone is None or not ws:
        return None
    return sum(1 for w in ws if _inside(w, zone, margin_px, frame_size)) / len(ws)


def _hand_to_hand(
    frame: PoseFrame, ws: Sequence[Point], s: float, desk: Optional[BBox],
    frame_size: tuple[int, int], cfg: PoseConfig,
) -> tuple[Optional[float], Optional[int]]:
    if desk is None or not ws:
        return None, None
    if not frame.others:
        return 0.0, None                       # в комнате больше никого: передачи точно нет
    best: Optional[float] = None
    best_track: Optional[int] = None
    margin = cfg.desk_margin_s * s
    for person in frame.others:
        for ow in wrists(person.kp, cfg.min_score):
            for cw in ws:
                mid = ((cw[0] + ow[0]) / 2, (cw[1] + ow[1]) / 2)
                value = (_closeness(dist_px(cw, ow, frame_size), cfg.hand_to_hand_s * s)
                         if _inside(mid, desk, margin, frame_size) else 0.0)
                if best is None or value > best:
                    best, best_track = value, person.track_id
    if best is None:
        return None, None                      # люди есть, но их кисти не видны
    return best, (best_track if best > 0 else None)


def _hand_at_face(kp: Optional[Keypoints], ws: Sequence[Point], s: float,
                  frame_size: tuple[int, int], cfg: PoseConfig) -> Optional[float]:
    head = head_point(kp, cfg.min_score)
    if head is None or not ws:
        return None
    return max(_closeness(dist_px(w, head, frame_size), cfg.face_s * s) for w in ws)


def _standing_by_legs(kp: Optional[Keypoints], frame_size: tuple[int, int], cfg: PoseConfig) -> Optional[float]:
    """Вертикальная проекция бедра против проекции торса (наклон камеры сокращает обе одинаково)."""
    h = frame_size[1]
    ratios: list[float] = []
    for sh_i, hip_i, knee_i, ankle_i in ((LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                                         (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)):
        sh, hip, knee = point(kp, sh_i, cfg.min_score), point(kp, hip_i, cfg.min_score), point(kp, knee_i, cfg.min_score)
        if sh is None or hip is None or knee is None:
            continue
        torso = (hip[1] - sh[1]) * h
        if torso <= 0:
            continue
        thigh = (knee[1] - hip[1]) * h
        ankle = point(kp, ankle_i, cfg.min_score)
        if ankle is not None and ankle[1] < knee[1]:
            thigh = min(thigh, 0.0)            # лодыжка выше колена — ноги согнуты, не стоит
        ratios.append(thigh / torso)
    if not ratios:
        return None
    return _ramp(max(ratios), cfg.standing_thigh_lo, cfg.standing_thigh_hi)


def frame_features(
    frame: PoseFrame,
    zones: Mapping[str, Optional[BBox]],
    frame_size: tuple[int, int],
    cfg: PoseConfig = PoseConfig(),
) -> dict[str, Any]:
    """Мгновенные признаки одного состояния (без окна пересчёта и запасного варианта «встал»).

    cashier_counting здесь — только «статическая» часть (кисти вместе над столом, 0/1),
    движение добавляет compute_features. Служебные ключи _s (масштаб) и _wrists (видимые
    запястья) нужны compute_features и в сцену не пишутся.
    """
    out: dict[str, Any] = {name: None for name in FEATURE_NAMES}
    out["hand_to_hand_track"] = None
    kp = frame.cashier_kp
    if kp is None:
        return out
    s = scale(kp, frame_size, frame.cashier_bbox, cfg.min_score)
    if s is None or s <= 0:
        return out
    ws = wrists(kp, cfg.min_score)
    desk, safe = zones.get("desk"), zones.get("safe")

    out["cashier_hand_in_safe"] = _hand_in_zone(ws, safe, cfg.safe_margin_s * s, frame_size)
    out["cashier_hands_on_desk"] = _hands_share(ws, desk, cfg.desk_margin_s * s, frame_size)
    h2h, track = _hand_to_hand(frame, ws, s, desk, frame_size, cfg)
    out["hand_to_hand"], out["hand_to_hand_track"] = h2h, track
    out["cashier_hand_at_face"] = _hand_at_face(kp, ws, s, frame_size, cfg)
    out["cashier_standing"] = _standing_by_legs(kp, frame_size, cfg)

    if desk is not None and len(ws) == 2:
        together = dist_px(ws[0], ws[1], frame_size) <= cfg.counting_hands_s * s
        on_desk = all(_inside(w, desk, cfg.desk_margin_s * s, frame_size) for w in ws)
        out["cashier_counting"] = 1.0 if together and on_desk else 0.0
    elif desk is not None and ws:
        out["cashier_counting"] = 0.0          # одна кисть видна: «обе вместе» не выполнено
    out["_s"] = s
    out["_wrists"] = ws
    return out


# --------------------------------------------------------------------------- признаки по ряду


def _motion(raw: Sequence[Mapping[str, Any]], times: Sequence[float], i: int,
            frame_size: tuple[int, int], cfg: PoseConfig) -> Optional[float]:
    """Путь обеих кистей кассира за окно вокруг состояния i, в единицах s (None — мало точек)."""
    half = cfg.counting_window_sec / 2
    idx = [j for j in range(len(raw)) if abs(times[j] - times[i]) <= half + 1e-6
           and len(raw[j].get("_wrists") or ()) == 2]
    if len(idx) < cfg.counting_min_points:
        return None
    path = 0.0
    for a, b in zip(idx, idx[1:]):
        wa = sorted(raw[a]["_wrists"])          # левая/правая по x: порядок не путается при перекрёсте
        wb = sorted(raw[b]["_wrists"])
        path += dist_px(wa[0], wb[0], frame_size) + dist_px(wa[1], wb[1], frame_size)
    s = median(raw[j]["_s"] for j in idx)
    span = times[idx[-1]] - times[idx[0]]
    if s <= 0 or span <= 0:
        return None
    return path / s * (cfg.counting_window_sec / span)   # к полному окну, если точки не на всю ширину


def compute_features(
    frames: Sequence[PoseFrame],
    zones: Mapping[str, Optional[BBox]],
    frame_size: tuple[int, int],
    cfg: PoseConfig = PoseConfig(),
) -> list[dict[str, Any]]:
    """Признаки по всем состояниям (порядок как во frames): окно пересчёта, запасной «встал»,
    медиана по 3 точкам. hand_to_hand_track не сглаживается (берётся с того же состояния)."""
    raw = [frame_features(f, zones, frame_size, cfg) for f in frames]
    times = [f.t for f in frames]

    counting: list[Optional[float]] = []
    for i, r in enumerate(raw):
        static = r["cashier_counting"]
        if static is None:
            counting.append(None)
            continue
        if static == 0.0 or (r["hand_to_hand"] is not None and r["hand_to_hand"] >= 0.5):
            counting.append(0.0)
            continue
        motion = _motion(raw, times, i, frame_size, cfg)
        # путь = порог -> 0,5; 1,5 порога и больше -> 1
        counting.append(None if motion is None else _clamp01(motion / cfg.counting_motion_s - 0.5))

    heights = [(f.cashier_bbox[3] - f.cashier_bbox[1]) for f in frames if f.cashier_bbox is not None]
    seated = median(heights) if heights else None
    standing: list[Optional[float]] = []
    for f, r in zip(frames, raw):
        value = r["cashier_standing"]
        if value is None and f.cashier_bbox is not None and seated:
            value = _ramp((f.cashier_bbox[3] - f.cashier_bbox[1]) / seated, cfg.standing_bbox_lo, cfg.standing_bbox_hi)
        standing.append(value)

    series: dict[str, list[Optional[float]]] = {
        "cashier_hand_in_safe": [r["cashier_hand_in_safe"] for r in raw],
        "cashier_hands_on_desk": [r["cashier_hands_on_desk"] for r in raw],
        "hand_to_hand": [r["hand_to_hand"] for r in raw],
        "cashier_counting": counting,
        "cashier_hand_at_face": [r["cashier_hand_at_face"] for r in raw],
        "cashier_standing": standing,
    }
    smoothed = {name: smooth3(vals, cfg.smooth_points) for name, vals in series.items()}
    out: list[dict[str, Any]] = []
    for i, r in enumerate(raw):
        row: dict[str, Any] = {name: _round(smoothed[name][i]) for name in FEATURE_NAMES}
        row["hand_to_hand_track"] = r["hand_to_hand_track"]
        out.append(row)
    return out
