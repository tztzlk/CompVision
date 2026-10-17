"""Состояния предметов без обучения: zero-shot-подписи SigLIP по вырезкам зон и классы COCO.

Для каждого состояния — зона, по вырезке которой оно оценивается, и две группы подписей:
позитивные («сейф открыт») и негативные («сейф закрыт»). Вероятность = доля softmax по всем
подписям состояния, пришедшаяся на позитивные. Подписи — первое приближение: их подстраивают
на машине с GPU по контактным листам scripts/viz_scene.py, модель при этом не обучается.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from worker.perception.geometry import BBox, expand

__all__ = [
    "StatePrompt",
    "STATE_PROMPTS",
    "STATE_PROMPTS_V2",
    "ZONE_CROP_MARGIN",
    "PHONE_LABELS",
    "FOOD_LABELS",
    "BAG_LABELS",
    "TABLE_LABELS",
    "COCO_LABELS_OF_INTEREST",
    "enabled_states",
    "crop_zone",
]


@dataclass(frozen=True)
class StatePrompt:
    zone: str                       # зона, по вырезке которой считается состояние
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    flag: Optional[str] = None      # имя флага SceneConfig; без него состояние не считается


_DESK_NEGATIVE = (
    "a photo of an empty office desk",
    "a photo of a desk with a laptop and a telephone",
    "a photo of a person working at a desk",
)

STATE_PROMPTS: dict[str, StatePrompt] = {
    "safe_open": StatePrompt(
        zone="safe",
        positive=("a photo of an open safe with its door open", "a photo of a metal cabinet with an open door"),
        negative=("a photo of a closed metal safe", "a photo of a closed grey metal cabinet"),
    ),
    "cash_on_desk": StatePrompt(
        zone="desk",
        positive=(
            "a photo of banknotes lying on a desk",
            "a photo of a stack of paper money on a table",
            "a photo of cash on an office desk",
        ),
        negative=_DESK_NEGATIVE,
    ),
    "bag_on_desk": StatePrompt(
        zone="desk",
        positive=(
            "a photo of a bag lying on a desk",
            "a photo of a handbag on a table",
            "a photo of a large fabric bag on a desk",
        ),
        negative=_DESK_NEGATIVE,
    ),
    "outerwear_on_desk": StatePrompt(
        zone="desk",
        positive=("a photo of a coat lying on a desk", "a photo of a jacket on a table"),
        negative=_DESK_NEGATIVE,
    ),
    "laptop_unlocked": StatePrompt(
        zone="laptop",
        positive=("a photo of a laptop screen showing a spreadsheet", "a photo of a laptop screen with an open document"),
        negative=(
            "a photo of a laptop showing a lock screen wallpaper",
            "a photo of a closed laptop",
            "a photo of a laptop with a black screen",
        ),
    ),
    # Стеклянная дверь и инкассаторская сумка zero-shot распознаются ненадёжно — по умолчанию выключены.
    "door_open": StatePrompt(
        zone="door",
        positive=("a photo of an open door",),
        negative=("a photo of a closed glass door",),
        flag="enable_door_state",
    ),
    "collection_bag": StatePrompt(
        zone="desk",
        positive=("a photo of a cash collection bag on a desk",),
        negative=_DESK_NEGATIVE,
        flag="enable_collection_bag",
    ),
}

# Вторая версия подписей (SceneConfig.state_prompts = "v2"): позитив и негатив — минимальные пары,
# различаются только наблюдаемым признаком («дверца открыта / закрыта», «на столе пальто / бумаги»).
# В v1 негатив «сейфа» отличался ещё цветом («grey») и словом (safe / cabinet), а среди негативов
# стола был «человек за столом»: когда кассир выходил из вырезки, вероятность перетекала
# к наличным, сумке и одежде. Для v2 людей в вырезке стола закрашивают (mask_people_in_desk).
_DESK_NEGATIVE_V2 = (
    "a photo of a desk with papers and a telephone on it",
    "a photo of a desk with nothing on it",
)
STATE_PROMPTS_V2: dict[str, StatePrompt] = {
    **STATE_PROMPTS,
    "safe_open": StatePrompt(
        zone="safe",
        positive=("a photo of a metal safe with its door open", "a photo of a metal cabinet with its door open"),
        negative=("a photo of a metal safe with its door closed", "a photo of a metal cabinet with its door closed"),
    ),
    "cash_on_desk": StatePrompt(
        zone="desk",
        positive=("a photo of a desk with banknotes on it", "a photo of a desk with a stack of paper money on it"),
        negative=_DESK_NEGATIVE_V2,
    ),
    "bag_on_desk": StatePrompt(
        zone="desk",
        positive=("a photo of a desk with a bag on it", "a photo of a desk with a handbag on it"),
        negative=_DESK_NEGATIVE_V2,
    ),
    "outerwear_on_desk": StatePrompt(
        zone="desk",
        positive=("a photo of a desk with a coat lying on it", "a photo of a desk with a jacket lying on it"),
        negative=_DESK_NEGATIVE_V2,
    ),
}

# Поля вырезки вокруг зоны (доля размера зоны с каждой стороны). Сейф — 0,25: открытая дверца
# выходит за рамку шкафа на ширину дверцы (замер 18.10, Х-20).
ZONE_CROP_MARGIN: dict[str, float] = {"desk": 0.10, "safe": 0.25, "laptop": 0.25, "door": 0.05}

# Классы COCO (RT-DETR), из которых считаются состояния без zero-shot.
PHONE_LABELS: tuple[str, ...] = ("cell phone",)
FOOD_LABELS: tuple[str, ...] = (
    "cup", "bottle", "wine glass", "bowl", "sandwich", "donut", "cake",
    "banana", "apple", "orange", "pizza", "hot dog",
)
BAG_LABELS: tuple[str, ...] = ("handbag", "backpack", "suitcase")
# Имена классов зависят от чекпойнта: в PekingU/rtdetr_r18vd_coco_o365 — "diningtable", "tvmonitor".
TABLE_LABELS: tuple[str, ...] = ("dining table", "diningtable")
COCO_LABELS_OF_INTEREST: tuple[str, ...] = (
    "person", *PHONE_LABELS, *FOOD_LABELS, *BAG_LABELS, *TABLE_LABELS,
    "laptop", "chair", "book", "keyboard", "tv", "tvmonitor",
)

_MIN_CROP_PX = 8


def enabled_states(config: Any) -> dict[str, StatePrompt]:
    """Состояния, включённые в SceneConfig (выключенные флагом — не считаются, остаются null).

    Набор подписей — по SceneConfig.state_prompts: "v1" (по умолчанию) или "v2" (минимальные пары).
    """
    prompts = STATE_PROMPTS_V2 if getattr(config, "state_prompts", "v1") == "v2" else STATE_PROMPTS
    return {
        name: spec for name, spec in prompts.items()
        if spec.flag is None or bool(getattr(config, spec.flag, False))
    }


def crop_zone(image: Any, bbox: BBox, margin: float) -> Optional[Any]:
    """Вырезка зоны из кадра (numpy HxWx3) с полями margin; None, если вырезка слишком мала."""
    h, w = image.shape[:2]
    box = expand(bbox, margin)
    x1, y1 = int(box[0] * w), int(box[1] * h)
    x2, y2 = int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None
    return image[y1:y2, x1:x2]
