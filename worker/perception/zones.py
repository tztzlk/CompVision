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

__all__ = ["ZONE_QUERIES", "DECOY_QUERIES", "COCO_FALLBACK", "safe_candidates", "pick_safe", "all_zone_queries", "infer_zones", "desk_candidates", "pick_desk_by_people"]

# Текстовые запросы OWLv2 для каждой зоны (английский — язык обучения модели).
ZONE_QUERIES: dict[str, tuple[str, ...]] = {
    "desk": ("a wooden office desk", "a desk with a computer"),
    "safe": ("a metal safe", "a grey metal cabinet", "a grey metal safe on the floor", "a small grey steel cabinet",
             "a safe"),
    "door": ("a glass door", "a door"),
    "laptop": ("a laptop",),
}
# Запросы-опознаватели: не зоны, а то, что OWLv2 путал с сейфом (счётная машинка и принтер на столе,
# Х-31, Х-43). Кандидат, который больше похож на них, чем на сейф, — не сейф.
DECOY_QUERIES: tuple[str, ...] = ("a printer", "a money counting machine")

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
    return list(dict.fromkeys([*(q for queries in ZONE_QUERIES.values() for q in queries), *DECOY_QUERIES]))


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
        and (det.bbox_xyxy[2] - det.bbox_xyxy[0]) * (det.bbox_xyxy[3] - det.bbox_xyxy[1]) >= _MIN_AREA["desk"]
    ]
    out = [(median_box(b for b, _ in cl), sum(s for _, s in cl)) for cl in _clusters(candidates)]
    return sorted(((b, s) for b, s in out if b is not None), key=lambda c: -c[1])


def pick_desk_by_people(candidates: Sequence[tuple[BBox, float]], person_boxes: Sequence[BBox],
                        margin: float = 0.15, min_share: float = 0.35) -> Optional[BBox]:
    """Рабочий стол — тот, у которого больше всего сидели и стояли люди, а не самый заметный
    (на переднем плане кадров кейса большой пустой белый стол). Для каждого кандидата —
    сумма долей рамок людей в расширенной рамке стола по всем кадрам; кандидат с долей меньше
    min_share от лучшего по OWLv2 не рассматривается (0,35: на Х-30 стол за стеклом с 0,26 от лучшего
    выигрывал — кассир стоял перед ним). None — людей нет или кандидат один."""
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


# Сейф (замер 18.10, контактный лист data/ml_results/zones_wrong.jpg): серый шкаф до пояса на полу или
# на подставке за рабочим местом либо сбоку от стола. OWLv2 выбирал вместо него счётную машинку и
# принтер на столе, белую тумбу перед столом и подставку под сейфом. Отбор кандидатов — геометрией:
_SAFE_MIN_SCORE = 0.1
_SAFE_MIN_KEYFRAMES = 2         # кандидат на меньшем числе ключевых кадров теряет в ранге пропорционально
_SAFE_HEIGHT = (0.3, 1.0)       # высота рамки в долях роста человека в кадре
_SAFE_ON_DESK = 0.3             # низ рамки ниже верхнего края стола больше чем на эту долю высоты стола...
_SAFE_DESK_OVERLAP = 0.3        # ...и рамка над столом по горизонтали хотя бы на эту долю — на столе или перед ним
_SAFE_DOOR_IOA = 0.6            # рамка почти вся в зоне двери — дверь или стекло
_SAFE_FLOOR = 0.05              # низ рамки выше верхнего края стола больше чем на это — шкаф за стеклом (Х-34, Х-37)
_SAFE_STACK = 0.6               # перекрытие по горизонтали, при котором рамка ниже другой — её подставка


def _h_overlap(a: BBox, b: BBox) -> float:
    """Доля ширины a, лежащая над/под b по горизонтали."""
    w = a[2] - a[0]
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / w if w > 0 else 0.0


def safe_candidates(owl_per_keyframe: Sequence[Sequence[Detection]], min_score: float = _SAFE_MIN_SCORE
                    ) -> list[dict[str, object]]:
    """Группы похожих рамок по запросам сейфа и опознавателей: медианная рамка, лучший score сейфа,
    лучший score опознавателя (принтер, машинка), число ключевых кадров с рамкой сейфа."""
    wanted = set(ZONE_QUERIES["safe"]) | set(DECOY_QUERIES)
    cands = [(det.bbox_xyxy, det.score, k, det.label) for k, dets in enumerate(owl_per_keyframe)
             for det in dets if det.label in wanted and det.score >= min_score]
    groups: list[list[tuple[BBox, float, int, str]]] = []
    for c in sorted(cands, key=lambda c: -c[1]):
        for g in groups:
            if iou(g[0][0], c[0]) >= _CLUSTER_IOU:
                g.append(c)
                break
        else:
            groups.append([c])
    out = []
    for g in groups:
        safe = [c for c in g if c[3] not in DECOY_QUERIES]
        if not safe:
            continue
        out.append({
            "bbox": median_box(c[0] for c in safe),
            "score": max(c[1] for c in safe),
            "decoy": max((c[1] for c in g if c[3] in DECOY_QUERIES), default=0.0),
            "keyframes": len({c[2] for c in safe}),
        })
    return sorted(out, key=lambda c: -float(c["score"]))  # type: ignore[arg-type]


def pick_safe(candidates: Sequence[dict[str, object]], desk: Optional[BBox], door: Optional[BBox],
              person_height: float) -> tuple[Optional[BBox], str]:
    """Сейф среди кандидатов safe_candidates: (рамка | None, почему так). Отбрасываются рамки на
    столешнице или перед столом (машинка, принтер, тумба), в зоне двери, не того размера относительно
    человека, больше похожие на принтер или машинку, и подставка, на которой стоит другой кандидат.
    Из оставшихся — с наибольшим score × min(1, ключевых кадров / _SAFE_MIN_KEYFRAMES)."""
    reasons: dict[str, int] = {}
    kept: list[dict[str, object]] = []
    for c in candidates:
        box: BBox = c["bbox"]  # type: ignore[assignment]
        h = box[3] - box[1]
        why = None
        if float(c["decoy"]) > float(c["score"]):  # type: ignore[arg-type]
            why = "decoy"
        elif person_height > 0 and not (_SAFE_HEIGHT[0] * person_height <= h <= _SAFE_HEIGHT[1] * person_height):
            why = "size"
        elif desk is not None and box[3] > desk[1] + _SAFE_ON_DESK * (desk[3] - desk[1]) \
                and _h_overlap(box, desk) >= _SAFE_DESK_OVERLAP:
            why = "on_desk"
        elif desk is not None and box[3] < desk[1] - _SAFE_FLOOR:
            why = "behind_glass"
        elif door is not None and ioa(box, door) >= _SAFE_DOOR_IOA:
            why = "door"
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            kept.append(c)

    def under_other(c: dict[str, object]) -> bool:
        b: BBox = c["bbox"]  # type: ignore[assignment]
        for o in kept:
            ob: BBox = o["bbox"]  # type: ignore[assignment]
            if o is c or _h_overlap(ob, b) < _SAFE_STACK:
                continue
            bh = b[3] - b[1]
            if ob[1] < b[1] - 0.3 * bh and b[1] - 0.1 * bh <= ob[3] <= b[1] + 0.5 * bh:
                return True       # другой кандидат стоит на этом (низ его — у верха этого): это подставка
        return False

    final = [c for c in kept if not under_other(c)]
    if not final:
        return None, "нет кандидата: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
    best = max(final, key=lambda c: float(c["score"]) * min(1.0, int(c["keyframes"]) / _SAFE_MIN_KEYFRAMES))  # type: ignore
    return best["bbox"], "ok"  # type: ignore[return-value]


def infer_zones(
    owl_per_keyframe: Sequence[Sequence[Detection]],
    coco_per_frame: Sequence[Sequence[Detection]],
    *,
    min_score: float = 0.12,
    coco_min_score: float = 0.5,
    coco_min_share: float = 0.2,
    person_boxes: Optional[Sequence[BBox]] = None,
    person_height: float = 0.0,
) -> tuple[dict[str, Optional[BBox]], dict[str, dict[str, object]]]:
    """Зоны и сведения о том, откуда они взяты: ({зона: рамка | None}, {зона: {source, score}}).

    person_boxes — рамки людей по всем кадрам: рабочий стол выбирается по ним (pick_desk_by_people).
    person_height — рост человека в кадре (доля высоты кадра) для отбора сейфа по размеру; 0 — без него.
    Сейф выбирается после стола и двери (pick_safe); не нашёлся — зоны нет, а в info причина.
    """
    keyframes = max(1, len(owl_per_keyframe))
    zones: dict[str, Optional[BBox]] = {}
    info: dict[str, dict[str, object]] = {}
    for zone, queries in ZONE_QUERIES.items():
        if zone == "safe":
            continue
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
    if person_boxes:
        desk = pick_desk_by_people(desk_candidates(owl_per_keyframe, min_score), person_boxes)
        if desk is not None and desk != zones.get("desk"):
            zones["desk"] = desk
            info["desk"] = {**info.get("desk", {}), "source": "owlv2+people"}
    cands = safe_candidates(owl_per_keyframe)
    safe, why = pick_safe(cands, zones.get("desk"), zones.get("door"), person_height)
    zones["safe"] = safe
    best = next((c for c in cands if c["bbox"] == safe), None)
    info["safe"] = ({"source": "owlv2", "score": round(float(best["score"]), 3)} if best is not None  # type: ignore[arg-type]
                    else {"source": None, "score": 0.0, "reason": why})
    return {name: zones.get(name) for name in ZONE_QUERIES}, info
