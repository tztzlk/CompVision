"""Контракт SceneTimeline — временной ряд состояний сцены (perception -> scene -> events).

Один JSON на видео: data/scene_cache/<video_id>.json (кэш для подбора порогов без GPU); тот же
объект в памяти возвращает worker.scene.build_scene. Пример — docs/scene_contract_example.json.

Модуль без внешних зависимостей (только stdlib): его импортируют worker (build_scene),
events/verdict и scripts/evaluate.py, в том числе на машине без torch и OpenCV.

Правила контракта (version 1):
  1. Вероятности 0..1; None (null) = «не удалось определить», не путать с 0 = «точно нет».
  2. Координаты — доли кадра 0..1, [x1, y1, x2, y2].
  3. states упорядочены по t (секунды от начала видео), шаг ≈ 1/fps_sampled.
  4. frame — JPEG относительно рабочего каталога задачи; в кэше кадров может не быть.
  5. cashier_present — вероятность, что кассир в зоне рабочего места (None — кассир не определён);
     persons_in_room — люди внутри кассы без людей за стеклом и отражений; others — остальные
     люди в комнате, first_seen_t — когда трек впервые появился внутри.
  6. person_entered / person_exited — в этот момент кто-то прошёл через зону двери внутрь / наружу.
  7. Поля можно добавлять: неизвестные поля состояния читаются в SceneState.extra, верхнего
     уровня — в SceneTimeline.extra, и сохраняются обратно. Переименовывать, удалять или менять
     смысл — только по согласию A и B с увеличением CONTRACT_VERSION.
"""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Optional

__all__ = [
    "CONTRACT_VERSION",
    "ZONE_NAMES",
    "PROB_FIELDS",
    "FLAG_FIELDS",
    "BBox",
    "SceneContractError",
    "Person",
    "SceneState",
    "SceneTimeline",
    "scene_from_dict",
    "scene_to_dict",
    "load_scene",
    "save_scene",
]

CONTRACT_VERSION = 1

ZONE_NAMES: tuple[str, ...] = ("desk", "workplace", "safe", "door", "laptop")

PROB_FIELDS: tuple[str, ...] = (
    "cashier_present",
    "safe_open",
    "door_open",
    "cash_on_desk",
    "bag_on_desk",
    "collection_bag",
    "outerwear_on_desk",
    "cup_or_food",
    "phone_in_hand",
    "laptop_unlocked",
)
FLAG_FIELDS: tuple[str, ...] = ("person_entered", "person_exited")

# Порядок полей состояния при записи — как в примере контракта.
_STATE_ORDER: tuple[str, ...] = (
    "t", "frame", "persons_in_room", "cashier_present", "cashier_bbox", "others",
    "person_entered", "person_exited", "safe_open", "door_open", "cash_on_desk", "bag_on_desk",
    "collection_bag", "outerwear_on_desk", "cup_or_food", "phone_in_hand", "laptop_unlocked",
)
_STATE_KNOWN = frozenset(_STATE_ORDER)
_TOP_REQUIRED: tuple[str, ...] = (
    "version", "video_id", "duration_sec", "fps_sampled", "frame_size", "zones", "states",
)
_TOP_KNOWN = frozenset(_TOP_REQUIRED) | {"meta"}

_COORD_TOL = 0.01   # допуск выхода координат за 0..1 (округление модели); дальше — ошибка
_PROB_TOL = 1e-6
_T_TOL_SEC = 1.0    # допуск t сверх duration_sec

BBox = tuple[float, float, float, float]


class SceneContractError(ValueError):
    """Данные не соответствуют контракту SceneTimeline."""


@dataclass(frozen=True)
class Person:
    """Человек в кассе (кроме кассира): трек, рамка, когда впервые появился внутри."""

    track_id: int
    bbox: BBox
    first_seen_t: float


@dataclass(frozen=True)
class SceneState:
    """Состояние сцены на момент t. Все вероятности: None = не определено."""

    t: float
    frame: str
    persons_in_room: Optional[int] = None
    cashier_present: Optional[float] = None
    cashier_bbox: Optional[BBox] = None
    others: tuple[Person, ...] = ()
    person_entered: Optional[bool] = False
    person_exited: Optional[bool] = False
    safe_open: Optional[float] = None
    door_open: Optional[float] = None
    cash_on_desk: Optional[float] = None
    bag_on_desk: Optional[float] = None
    collection_bag: Optional[float] = None
    outerwear_on_desk: Optional[float] = None
    cup_or_food: Optional[float] = None
    phone_in_hand: Optional[float] = None
    laptop_unlocked: Optional[float] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def value(self, name: str) -> Any:
        """Значение поля по имени, включая добавленные поля из extra (None, если поля нет)."""
        if name in _STATE_KNOWN:
            return getattr(self, name)
        return self.extra.get(name)


@dataclass(frozen=True)
class SceneTimeline:
    """Временной ряд состояний сцены одного видео."""

    video_id: str
    duration_sec: float
    fps_sampled: float
    frame_size: tuple[int, int]                  # ширина, высота исходного видео
    zones: Mapping[str, Optional[BBox]]
    states: tuple[SceneState, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)
    version: int = CONTRACT_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def times(self) -> list[float]:
        return [s.t for s in self.states]

    def series(self, name: str) -> list[tuple[float, Any]]:
        """[(t, значение поля name)] по всем состояниям — удобно для правил в events."""
        return [(s.t, s.value(name)) for s in self.states]

    def state_at(self, t: float) -> Optional[SceneState]:
        """Ближайшее по времени состояние (None, если состояний нет)."""
        if not self.states:
            return None
        times = self.times()
        i = bisect_left(times, t)
        if i <= 0:
            return self.states[0]
        if i >= len(times):
            return self.states[-1]
        before, after = self.states[i - 1], self.states[i]
        return before if t - before.t <= after.t - t else after


# --------------------------------------------------------------------------- разбор и проверка


def _fail(where: str, message: str) -> NoReturn:
    raise SceneContractError(f"{where}: {message}")


def _number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, f"ожидается число, получено {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        _fail(where, "число должно быть конечным")
    return number


def _prob(where: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    number = _number(where, value)
    if number < -_PROB_TOL or number > 1.0 + _PROB_TOL:
        _fail(where, f"вероятность вне 0..1: {number}")
    return min(1.0, max(0.0, number))


def _bbox(where: str, value: Any) -> Optional[BBox]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        _fail(where, "ожидается [x1, y1, x2, y2]")
    coords = [_number(f"{where}[{i}]", v) for i, v in enumerate(value)]
    if any(c < -_COORD_TOL or c > 1.0 + _COORD_TOL for c in coords):
        _fail(where, f"координаты вне 0..1: {coords}")
    x1, y1, x2, y2 = (min(1.0, max(0.0, c)) for c in coords)
    if x2 < x1 or y2 < y1:
        _fail(where, f"ожидается x1 <= x2 и y1 <= y2: {coords}")
    return (x1, y1, x2, y2)


def _flag(where: str, value: Any) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    _fail(where, "ожидается true, false или null")


def _count(where: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(where, "ожидается целое число >= 0 или null")
    return value


def _person(where: str, raw: Any) -> Person:
    if not isinstance(raw, dict):
        _fail(where, "ожидается объект {track_id, bbox, first_seen_t}")
    for key in ("track_id", "bbox", "first_seen_t"):
        if key not in raw:
            _fail(where, f"нет поля {key}")
    track_id = raw["track_id"]
    if isinstance(track_id, bool) or not isinstance(track_id, int):
        _fail(f"{where}.track_id", "ожидается целое число")
    bbox = _bbox(f"{where}.bbox", raw["bbox"])
    if bbox is None:
        _fail(f"{where}.bbox", "рамка человека обязательна")
    return Person(track_id=track_id, bbox=bbox, first_seen_t=_number(f"{where}.first_seen_t", raw["first_seen_t"]))


def _state(where: str, raw: Any) -> SceneState:
    if not isinstance(raw, dict):
        _fail(where, "ожидается объект состояния")
    for key in ("t", "frame"):
        if key not in raw:
            _fail(where, f"нет обязательного поля {key}")
    t = _number(f"{where}.t", raw["t"])
    if t < 0:
        _fail(f"{where}.t", "время не может быть отрицательным")
    frame = raw["frame"]
    if not isinstance(frame, str):
        _fail(f"{where}.frame", "ожидается строка (путь к JPEG относительно work_dir)")
    others_raw = raw.get("others") or []
    if not isinstance(others_raw, list):
        _fail(f"{where}.others", "ожидается список")
    return SceneState(
        t=t,
        frame=frame,
        persons_in_room=_count(f"{where}.persons_in_room", raw.get("persons_in_room")),
        cashier_bbox=_bbox(f"{where}.cashier_bbox", raw.get("cashier_bbox")),
        others=tuple(_person(f"{where}.others[{i}]", p) for i, p in enumerate(others_raw)),
        **{name: _flag(f"{where}.{name}", raw.get(name, False)) for name in FLAG_FIELDS},
        **{name: _prob(f"{where}.{name}", raw.get(name)) for name in PROB_FIELDS},
        extra={k: v for k, v in raw.items() if k not in _STATE_KNOWN},
    )


def scene_from_dict(data: Any, where: str = "scene") -> SceneTimeline:
    """Разобрать и проверить словарь (из JSON) по контракту. Ошибки — SceneContractError."""
    if not isinstance(data, dict):
        _fail(where, "ожидается объект JSON")
    missing = [key for key in _TOP_REQUIRED if key not in data]
    if missing:
        _fail(where, f"нет обязательных полей {missing}")

    version = data["version"]
    if isinstance(version, bool) or version != CONTRACT_VERSION:
        _fail(f"{where}.version", f"поддерживается только {CONTRACT_VERSION}, получено {version!r}")
    video_id = data["video_id"]
    if not isinstance(video_id, str) or not video_id.strip():
        _fail(f"{where}.video_id", "ожидается непустая строка")
    duration = _number(f"{where}.duration_sec", data["duration_sec"])
    if duration < 0:
        _fail(f"{where}.duration_sec", "длительность не может быть отрицательной")
    fps = _number(f"{where}.fps_sampled", data["fps_sampled"])
    if fps <= 0:
        _fail(f"{where}.fps_sampled", "частота должна быть > 0")
    size = data["frame_size"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        _fail(f"{where}.frame_size", "ожидается [ширина, высота] — целые > 0")

    zones_raw = data["zones"]
    if not isinstance(zones_raw, dict):
        _fail(f"{where}.zones", "ожидается объект {имя: [x1, y1, x2, y2] | null}")
    zones: dict[str, Optional[BBox]] = {name: None for name in ZONE_NAMES}
    for name, box in zones_raw.items():
        if not isinstance(name, str):
            _fail(f"{where}.zones", "имена зон — строки")
        zones[name] = _bbox(f"{where}.zones.{name}", box)

    states_raw = data["states"]
    if not isinstance(states_raw, list):
        _fail(f"{where}.states", "ожидается список")
    states = tuple(_state(f"{where}.states[{i}]", s) for i, s in enumerate(states_raw))
    for i in range(1, len(states)):
        if states[i].t < states[i - 1].t:
            _fail(f"{where}.states[{i}].t", "состояния должны идти по возрастанию t")
    if states and states[-1].t > duration + _T_TOL_SEC:
        _fail(f"{where}.states", f"t={states[-1].t} больше длительности {duration}")

    meta = data.get("meta") or {}
    if not isinstance(meta, dict):
        _fail(f"{where}.meta", "ожидается объект")
    return SceneTimeline(
        video_id=video_id,
        duration_sec=duration,
        fps_sampled=fps,
        frame_size=(int(size[0]), int(size[1])),
        zones=zones,
        states=states,
        meta=dict(meta),
        version=CONTRACT_VERSION,
        extra={k: v for k, v in data.items() if k not in _TOP_KNOWN},
    )


# --------------------------------------------------------------------------- запись


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _round_box(box: Optional[BBox]) -> Optional[list[float]]:
    return None if box is None else [round(float(c), 4) for c in box]


def _state_to_dict(state: SceneState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _STATE_ORDER:
        value = getattr(state, name)
        if name == "t":
            out[name] = round(float(value), 3)
        elif name == "cashier_bbox":
            out[name] = _round_box(value)
        elif name == "others":
            out[name] = [
                {"track_id": p.track_id, "bbox": _round_box(p.bbox), "first_seen_t": round(float(p.first_seen_t), 3)}
                for p in value
            ]
        elif name in PROB_FIELDS:
            out[name] = _round(value, 3)
        else:
            out[name] = value
    for key, value in state.extra.items():
        if key not in out:
            out[key] = value
    return out


def scene_to_dict(scene: SceneTimeline) -> dict[str, Any]:
    """SceneTimeline -> словарь для JSON (порядок полей как в примере контракта)."""
    zones = {name: _round_box(scene.zones.get(name)) for name in ZONE_NAMES}
    zones.update({name: _round_box(box) for name, box in scene.zones.items() if name not in zones})
    out: dict[str, Any] = {
        "version": scene.version,
        "video_id": scene.video_id,
        "duration_sec": round(float(scene.duration_sec), 3),
        "fps_sampled": round(float(scene.fps_sampled), 3),
        "frame_size": [int(scene.frame_size[0]), int(scene.frame_size[1])],
        "zones": zones,
        "states": [_state_to_dict(s) for s in scene.states],
        "meta": dict(scene.meta),
    }
    for key, value in scene.extra.items():
        if key not in out:
            out[key] = value
    return out


def load_scene(path: Path) -> SceneTimeline:
    """Прочитать и проверить scene.json."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneContractError(f"не удалось прочитать {p}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SceneContractError(f"{p}: некорректный JSON: {exc}") from exc
    return scene_from_dict(data, where=str(p))


def save_scene(scene: SceneTimeline, path: Path) -> Path:
    """Записать scene.json атомарно (через временный файл). Невалидный объект не пишется."""
    data = scene_to_dict(scene)
    scene_from_dict(data, where=f"save_scene({scene.video_id})")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p
