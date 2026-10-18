"""Сборка SceneTimeline из треков, детекций, зон и zero-shot-оценок (чистая логика, только stdlib).

Модели запускает worker.scene.pipeline; здесь — правила восприятия сцены:
  - склейка разорванных треков и отбрасывание коротких;
  - «внутри кассы / за стеклом» по высоте рамки и положению относительно стола;
  - роль кассира без распознавания лиц: трек, дольше всех и ближе всех находящийся у стола;
  - зона рабочего места (workplace) по положениям кассира у стола;
  - вход и выход через зону двери;
  - состояния предметов: классы COCO у рук кассира и на столе + zero-shot-оценки по зонам.
Все пороги — в SceneConfig; их подбирают на машине с GPU по контактным листам scripts/viz_scene.py.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import asdict, dataclass, field, replace
from statistics import median
from typing import Any, Mapping, Optional, Sequence

from worker.perception import Detection
from worker.perception.geometry import (
    BBox,
    area,
    center,
    center_dist,
    contains_point,
    expand,
    height,
    ioa,
    iou,
    median_box,
    pad,
)
from worker.perception.states import BAG_LABELS, FOOD_LABELS, PHONE_LABELS
from worker.perception.tracking import Track, TrackPoint
from worker.scene.contract import ZONE_NAMES, Person, SceneState, SceneTimeline

__all__ = ["SceneConfig", "FrameRef", "stitch_tracks", "assemble_timeline", "with_brightness", "with_door_change",
           "ITEM_QUERIES", "item_check_times", "with_left_items", "track_height",
           "SCAN_QUERIES", "DESK_ITEM_GROUPS", "with_desk_items", "with_ft_items", "people_at"]


@dataclass(frozen=True)
class SceneConfig:
    """Параметры восприятия сцены (доли кадра, секунды, вероятности)."""

    # кадры и детектор
    fps: float = 3.0                    # частота анализа кадров
    batch_size: int = 8                 # кадров на один прогон RT-DETR
    jpeg_quality: int = 90
    person_score: float = 0.35          # минимальный score человека
    object_score: float = 0.30          # минимальный score предмета COCO
    person_nms_iou: float = 0.7         # подавление дублей рамок человека
    # трекер
    track_iou: float = 0.2
    track_center_dist: float = 0.12
    track_max_lost_sec: float = 1.5
    track_new_score: float = 0.5
    stitch_gap_sec: float = 1.0         # трек, пропавший не дольше, склеивается со следующим
    stitch_center_dist: float = 0.15
    # Трек, появившийся далеко от двери, — тот же человек, которого перекрыли (клиента за кассиром,
    # кассира у сейфа): склеивается с треком, потерянным на том же месте, и при разрыве до этого.
    stitch_far_gap_sec: float = 8.0
    stitch_far_iou: float = 0.3
    min_track_points: int = 2
    # внутри кассы / за стеклом
    inside_height_ratio: float = 0.5    # высота рамки >= доли типичной высоты человека в кассе
    floor_margin: float = 0.0           # низ рамки ниже верхнего края стола (+ запас) — человек в комнате
    # Замер 17.10: когда кассир сидит лицом к камере у дальней стены, он сам мелкий, и «рамка не
    # меньше половины кассира» пропускала людей за стеклом. При найденном столе внутри только тот,
    # чей низ рамки не выше верхнего края стола больше чем на floor_tolerance (доля кадра).
    floor_required: bool = True
    floor_tolerance: float = 0.03
    hold_gap_sec: float = 1.0           # пропуск детекции внутри трека не дольше — человек на месте
    # кассир и рабочее место
    desk_margin: float = 0.15           # расширение зоны стола при поиске кассира и предметов
    workplace_min_desk_ioa: float = 0.15
    workplace_margin: float = 0.05
    presence_ioa: float = 0.5           # доля рамки кассира в рабочей зоне, при которой presence = 1
    cashier_extra_ioa: float = 0.4      # трек вне времени основного кассира на его месте — тоже кассир
    cashier_overlap_sec: float = 1.0    # ...даже если пересекается с ним по времени не дольше (разрыв трека, Х-35)
    # Признаки роли кассира сверх «близость к столу × размер» (размер обманывает, когда клиентский
    # стул ближе к камере, чем кассир, — как в Х-44): к сейфу подходит кассир, клиент — нет.
    cashier_safe_bonus: float = 2.0     # множитель счёта трека, который подходил к сейфу
    cashier_safe_ioa: float = 0.15      # доля рамки в зоне сейфа, при которой человек «у сейфа»
    cashier_start_bonus: float = 1.25   # множитель трека, который в кассе с первой секунды
    # вход / выход
    door_margin: float = 0.08
    door_min_ioa: float = 0.1
    entry_grace_sec: float = 0.7        # кто в кадре с самого начала — не «вошёл»
    exit_grace_sec: float = 0.7
    min_event_track_sec: float = 0.6    # трек внутри короче — не порождает вход/выход
    # Что считать входом через дверь (замер 17.10: «room» оставил входы в 4 видео из 42 — люди,
    # вставшие у края стола рядом с дверью, не выходят из её расширенной зоны):
    #   "stay" — первая точка у двери и человек пробыл внутри не меньше min_room_sec;
    #   "move" — то же, плюс центр рамки сместился от точки входа хотя бы на entry_move × высоту рамки;
    #   "room" — то же, плюс человек хотя бы min_room_sec вне зоны двери.
    entry_mode: str = "stay"
    min_room_sec: float = 1.0
    entry_move: float = 0.5
    # Уход с рабочего места (замер 18.10, Х-15, Х-24, Х-26: кассир выходит в дверь и идёт по коридору
    # за стеклянной перегородкой; трек продолжается, рамка за стеклом снова накрывает рабочее место
    # на 1–2 с, и cashier_present не падал). Уход: кассир отошёл от рабочего места (центр рамки дальше
    # leave_far × ширины рабочего места, не у сейфа) и не вернулся — нет leave_return_sec подряд у места
    # (ближе leave_home) с уверенной детекцией (score >= leave_return_score: за стеклом 0,4–0,85).
    # Таймкод — последний кадр у места; если кассир перед этим встал (рамка выросла в leave_up_ratio
    # раз против сидячей) не раньше чем за leave_up_sec, — момент, когда встал.
    leave_home: float = 0.35
    leave_far: float = 0.6
    leave_return_sec: float = 2.0
    leave_return_score: float = 0.85
    leave_min_away_sec: float = 3.0
    leave_up_ratio: float = 1.25
    leave_up_sec: float = 3.0
    # Что осталось на столе после ухода: OWLv2 по ITEM_QUERIES на кадрах, где кассира уже нет у места
    # (через item_check_sec после первого кадра вдали), — SigLIP по всей вырезке стола мелкую пачку
    # денег не видит (на правильном столе cash_on_desk <= 0,004 и там, где деньги на кадре).
    item_check_sec: tuple[float, ...] = (0.7, 2.0)
    # Сканирование стола OWLv2 раз в item_scan_sec (worker.scene.pipeline.scan_desk_items): поля
    # desk_cash, desk_cup, desk_coat, desk_collection_bag — score лучшего предмета группы в зоне стола,
    # приведённый к 0..1: (score - item_scan_floor) / item_scan_span. 0 — сканирование выключено.
    item_scan_sec: float = 2.0
    item_scan_max: int = 80             # не больше стольких кадров скана (Х-14, 142 с: 71 кадр — 51 с на GPU)
    item_scan_floor: float = 0.1
    item_scan_span: float = 0.3
    # предметы
    hand_margin: float = 0.15           # расширение рамки кассира: «предмет в руке»
    # зоны и zero-shot
    zone_keyframes: int = 4
    zone_min_score: float = 0.12
    # Замер 18.10: сейф часто не в тройке лучших рамок OWLv2 (её занимают тумба и счётная машинка) —
    # кандидатов больше, отбор по геометрии в worker.perception.zones.
    zone_owl_score: float = 0.05
    zone_top_k: int = 6
    state_every_sec: float = 1.0
    state_batch_frames: int = 16
    enable_door_state: bool = False
    enable_collection_bag: bool = False
    # Дверь не в том виде, что в начале ролика (R15, worker.perception.door): полоса зоны двери от
    # door_band_top до door_band_bottom её высоты (ниже — стул клиента), эталон — первые door_baseline_sec
    # секунд, кадр без оценки, если люди закрывают больше door_max_masked полосы. В сцену — door_diff
    # (сырое расхождение) и door_change = (door_diff - door_diff_floor) / door_diff_span в 0..1.
    # Замер 18.10 по кадрам: дверь открыта (Х-32, Х-33) 0,77–0,99; закрыта после прохода 0,03–0,2.
    door_change: bool = True
    door_band_top: float = 0.0
    door_band_bottom: float = 0.65
    door_baseline_sec: float = 1.0
    door_max_masked: float = 0.5
    door_person_margin: float = 0.02
    door_diff_floor: float = 0.35
    door_diff_span: float = 0.3
    # Zero-shot: набор подписей ("v1" | "v2", worker.perception.states) и закрашивание людей
    # в вырезке стола перед SigLIP (спина или торс кассира в вырезке путают «одежду на столе»).
    # Замер 18.10 (кэш r2, правильные зоны): v2 + маска — 8 нарушений из 21 при 0 ложных, v1 — 6 и 0.
    state_prompts: str = "v2"
    # Стол выбирать по людям (где сидели и стояли), а не по самому заметному кандидату OWLv2
    # (Х-20, Х-22: иначе стол — полоса за стеклом в соседнем офисе).
    desk_by_people: bool = True
    mask_people_in_desk: bool = True
    # Голова «сейф открыт» (worker.perception.heads, scripts/train_heads.py): путь к .npz относительно
    # каталога моделей. Пусто — safe_open считает zero-shot SigLIP; нет файла — тоже zero-shot
    # (предупреждение в meta.safe_head.error). С головой zero-shot-значение остаётся в state_samples
    # под именем safe_open_zero_shot. Замер 18.10 (вне фолда, кэши r3 и аугментаций): на оригинале и
    # flip без изменений, на persp возвращает Х-20 (R10), ложных 0 — включена по умолчанию.
    safe_head: str = "heads/safe_open.npz"
    # Дообученная башня SigLIP (worker.perception.ft_tower, scripts/finetune_siglip.py final) — вторая,
    # отдельная от общего SigLIP (тот не подменяется): каталог весов относительно каталога моделей
    # (в контейнере /models/ft/siglip_states; state.pt — последние 4 блока, post_layernorm, pooling и
    # голова, fp16; meta.json). Нет весов или ошибка — сцена как без башни (предупреждение в
    # meta.safe_ft.error). Пусто — выключена.
    safe_ft: str = "ft/siglip_states"
    # Какие задачи башни идут в сцену: desk_cash — поле desk_cash_ft только для R14 (desk_cash OWLv2 не
    # меняется; R14 при нём — ft_* и «пустое окно» в rules.yaml); safe_open — вместо головы safe_head
    # (значение головы остаётся в state_samples.safe_open_head; R10 — ft_* поверх head_*).
    # Замер 18.10 (вне фолда на всех секундах, кэш r3 и 4 аугментации): desk_cash + «пустое окно» —
    # Х-4 ловится на flip/crop/light (было только на оригинале), ложных 0, таймкод не хуже ни на одном
    # варианте; safe_open башни сдвигает R10 на Х-20 на 1 с раньше (persp/crop: вне ±3 с) — не включён.
    safe_ft_fields: tuple[str, ...] = ("desk_cash",)
    # Голова «человек внутри кассы» (worker.perception.heads, scripts/train_heads.py person_inside):
    # путь к .npz относительно каталога моделей; пусто — внутри/за стеклом решает только геометрия.
    # Голова решает лишь спорные точки: геометрия «за стеклом», но низ рамки не выше верха стола больше
    # чем на inside_head_band, рамка не мельче inside_height_ratio роста и голова >= inside_head_hi —
    # человек в комнате (Х-33: посторонний за спиной кассира, ноги закрыты столом); геометрия «внутри»,
    # а голова <= inside_head_lo — за стеклом (кассир в коридоре после ухода); в зоне двери вторую
    # правку не делаем — голова путает проём со стеклом, и терялся вход (Х-38 на flip).
    # Замер 18.10: ошибок «внутри/за стеклом» по точкам 63 -> 30 (вне фолда), но по правилам без
    # изменений на оригинале и всех аугментациях, Х-33 голова не ловит (0,13–0,17) — выключена.
    inside_head: str = ""
    inside_head_hi: float = 0.9
    inside_head_lo: float = 0.1
    inside_head_band: float = 0.08
    inside_head_max_dt: float = 0.7     # оценка головы берётся с ближайшей точки трека не дальше, с
    # Дообученная башня «человек внутри кассы» (scripts/finetune_siglip.py person-final, worker.perception.ft_tower):
    # каталог весов относительно каталога моделей (models/ft/person_inside: последние 4 блока SigLIP, pooling и
    # голова на [эмбеддинг вырезки crop_person, person_geo]). Пусто — выключена. Оценка — на каждой точке трека
    # (сырой кэш: person_ft). Правит точки в обе стороны: p < inside_ft_lo — за стеклом (кассир в коридоре над
    # рабочим местом не «у места», не возвращение, не в persons_in_room; выход из кассы — последняя точка внутри
    # у двери), геометрия «за стеклом», но p >= inside_ft_hi и рамка не мельче inside_height_ratio роста и низом
    # не выше верха стола больше чем на inside_head_band — внутри. В зоне двери правка «за стеклом» — только
    # при inside_ft_door (проём размечен «?», башня его не училась; на Х-38 так терялся вход).
    inside_ft: str = ""
    inside_ft_lo: float = 0.2
    inside_ft_hi: float = 0.9
    inside_ft_door: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameRef:
    index: int          # позиция кадра в анализе (совпадает с TrackPoint.frame_index)
    t: float
    frame: str          # JPEG относительно work_dir, например "frames/000051.jpg"


@dataclass(frozen=True)
class _Entry:
    bbox: BBox
    inside: bool
    score: float = 1.0


@dataclass
class _TrackInfo:
    track: Track
    inside: list[bool]
    by_frame: dict[int, _Entry] = field(default_factory=dict)
    role: str = "other"                 # "cashier" | "other"

    def inside_points(self) -> list[TrackPoint]:
        return [p for p, ins in zip(self.track.points, self.inside) if ins]


# --------------------------------------------------------------------------- треки


def stitch_tracks(tracks: Sequence[Track], config: SceneConfig, door: Optional[BBox] = None) -> list[Track]:
    """Склеить трек с более ранним, если тот пропал не дольше stitch_gap_sec назад рядом.

    Трек, который начинается далеко от двери (door — зона двери; None — правило не действует),
    склеивается и с треком, потерянным до stitch_far_gap_sec назад на том же месте (IoU рамок):
    люди не появляются посреди комнаты — это тот же человек после перекрытия.
    Возвращает новые объекты Track (входные не меняются); id склеенного — id более раннего.
    """
    door_zone = pad(door, config.door_margin) if door else None
    ordered = sorted((Track(t.track_id, list(t.points)) for t in tracks if t.points), key=lambda tr: tr.first_t)
    merged: list[Track] = []
    for track in ordered:
        best: Optional[Track] = None
        best_dist = math.inf
        far_from_door = door_zone is not None and not _near_door(track.points[0].bbox, door_zone, config)
        for candidate in merged:
            gap = track.first_t - candidate.last_t
            if gap <= 0:
                continue
            last, first = candidate.points[-1].bbox, track.points[0].bbox
            dist = center_dist(last, first)
            near = gap <= config.stitch_gap_sec and dist <= config.stitch_center_dist
            same_place = (far_from_door and gap <= config.stitch_far_gap_sec
                          and (iou(last, first) >= config.stitch_far_iou or dist <= config.stitch_center_dist))
            if (near or same_place) and dist < best_dist:
                best, best_dist = candidate, dist
        if best is None:
            merged.append(track)
        else:
            best.points.extend(track.points)
    return merged


def track_height(tracks: Sequence[Track]) -> float:
    """Рост человека в кадре: наибольшая медианная высота рамки среди треков длиннее 1 с."""
    heights = []
    for tr in tracks:
        if tr.points and tr.points[-1].t - tr.points[0].t >= 1.0:
            hs = sorted(p.bbox[3] - p.bbox[1] for p in tr.points)
            heights.append(hs[len(hs) // 2])
    return max(heights, default=0.0)


def _reference_height(tracks: Sequence[Track]) -> float:
    """Типичная высота человека внутри кассы: наибольшая медианная высота среди долгих треков."""
    long_tracks = [t for t in tracks if t.duration >= 1.0] or list(tracks)
    if not long_tracks:
        return 0.0
    return max(median(height(p.bbox) for p in t.points) for t in long_tracks)


def _head_key(frame_index: int, bbox: Sequence[float]) -> tuple:
    return (int(frame_index), *(round(float(v), 3) for v in bbox))


def _inside_flags(track: Track, ref_height: float, desk: Optional[BBox], config: SceneConfig,
                  door: Optional[BBox] = None, head: Optional[Mapping[tuple, float]] = None,
                  ft: Optional[Mapping[tuple, float]] = None) -> list[bool]:
    """Внутри кассы ли человек в каждой точке трека (сглажено большинством по 3 точкам).

    Люди за стеклом и отражения — мелкие рамки, низ которых выше верхнего края стола.
    Человек, который весь трек остаётся в зоне стеклянной двери и ни разу не становится крупным,
    виден сквозь стекло (door — зона двери): он целиком снаружи.
    head — вероятности головы «внутри кассы» по точкам (_head_key): правят спорные точки (SceneConfig.inside_head).
    ft — вероятности дообученной башни по точкам (_head_key, SceneConfig.inside_ft): после головы, см. _apply_ft.
    """
    if door is not None and ref_height > 0:
        door_zone = pad(door, config.door_margin)
        small = all(height(p.bbox) < config.inside_height_ratio * ref_height for p in track.points)
        if small and all(_near_door(p.bbox, door_zone, config) for p in track.points):
            return [False] * len(track.points)
    raw = []
    for point in track.points:
        by_height = ref_height > 0 and height(point.bbox) >= config.inside_height_ratio * ref_height
        by_floor = desk is not None and point.bbox[3] >= desk[1] + config.floor_margin
        if desk is not None and config.floor_required:
            raw.append(point.bbox[3] >= desk[1] - config.floor_tolerance)
        else:
            raw.append(by_height or by_floor)
    if head:
        door_zone = pad(door, config.door_margin) if door is not None else None
        probs = [(p.t, head.get(_head_key(p.frame_index, p.bbox))) for p in track.points]
        known = [(t, v) for t, v in probs if v is not None]
        for i, point in enumerate(track.points):
            near = min(known, key=lambda tv: abs(tv[0] - point.t), default=None)
            if near is None or abs(near[0] - point.t) > config.inside_head_max_dt:
                continue
            ph = near[1]
            if not raw[i] and ph >= config.inside_head_hi and desk is not None \
                    and point.bbox[3] >= desk[1] - config.inside_head_band \
                    and ref_height > 0 and height(point.bbox) >= config.inside_height_ratio * ref_height:
                raw[i] = True
            elif raw[i] and ph <= config.inside_head_lo and not (door_zone and _near_door(point.bbox, door_zone, config)):
                # в зоне двери не правим: там входят и выходят, а голова путает проём со стеклом (Х-38 flip)
                raw[i] = False
    if ft:
        raw = _apply_ft(track, raw, ref_height, desk, config, door, ft)
    smoothed = []
    for i, value in enumerate(raw):
        window = raw[max(0, i - 1): i + 2]
        votes = sum(window) * 2
        smoothed.append(value if votes == len(window) else votes > len(window))
    return smoothed


def _apply_ft(track: Track, raw: list[bool], ref_height: float, desk: Optional[BBox], config: SceneConfig,
              door: Optional[BBox], ft: Mapping[tuple, float]) -> list[bool]:
    """Флаги «внутри» по вероятностям башни inside_ft (точка без оценки — по ближайшей не дальше
    inside_head_max_dt): p < inside_ft_lo — за стеклом (в зоне двери — только при inside_ft_door),
    геометрия «за стеклом», p >= inside_ft_hi, рамка не мельче inside_height_ratio роста и низом у стола — внутри."""
    door_zone = pad(door, config.door_margin) if door is not None else None
    probs = [(p.t, ft.get(_head_key(p.frame_index, p.bbox))) for p in track.points]
    known = [(t, v) for t, v in probs if v is not None]
    out = list(raw)
    for i, point in enumerate(track.points):
        pf = probs[i][1]
        if pf is None:
            near = min(known, key=lambda tv: abs(tv[0] - point.t), default=None)
            if near is None or abs(near[0] - point.t) > config.inside_head_max_dt:
                continue
            pf = near[1]
        if out[i] and pf < config.inside_ft_lo:
            if config.inside_ft_door or not (door_zone and _near_door(point.bbox, door_zone, config)):
                out[i] = False
        elif not out[i] and pf >= config.inside_ft_hi and desk is not None \
                and point.bbox[3] >= desk[1] - config.inside_head_band \
                and ref_height > 0 and height(point.bbox) >= config.inside_height_ratio * ref_height:
            out[i] = True
    return out


def _fill_frames(info: _TrackInfo, times: Sequence[float], config: SceneConfig) -> None:
    """Сопоставить трек кадрам; короткие пропуски детекции (<= hold_gap_sec) заполнить ближайшей точкой."""
    points = info.track.points
    for point, inside in zip(points, info.inside):
        info.by_frame[point.frame_index] = _Entry(point.bbox, inside, point.score)
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if b.t - a.t > config.hold_gap_sec:
            continue
        for k in range(a.frame_index + 1, b.frame_index):
            nearer_a = times[k] - a.t <= b.t - times[k]
            source, inside = (a, info.inside[i]) if nearer_a else (b, info.inside[i + 1])
            info.by_frame[k] = _Entry(source.bbox, inside, source.score)


# --------------------------------------------------------------------------- роли и события


def _select_cashier(infos: Sequence[_TrackInfo], desk: Optional[BBox], config: SceneConfig,
                    safe: Optional[BBox] = None) -> Optional[_TrackInfo]:
    """Кассир — трек с наибольшим счётом: сумма «близость к столу × размер» по точкам внутри кассы,
    умноженная на признаки роли.

    Размер помогает, когда кассир сидит спиной к камере, а клиент напротив. Но если клиентский стул
    ближе к камере (Х-44), размер выбрал бы клиента, поэтому счёт трека, который подходил к сейфу,
    умножается на cashier_safe_bonus, а трека, который в кассе с первой секунды, — на cashier_start_bonus.
    """
    desk_zone = expand(desk, config.desk_margin) if desk else None
    safe_zone = expand(safe, config.desk_margin) if safe else None
    best: Optional[_TrackInfo] = None
    best_score = 0.0
    for info in infos:
        score = 0.0
        at_safe = False
        for point, inside in zip(info.track.points, info.inside):
            if not inside:
                continue
            weight = math.sqrt(area(point.bbox))
            if desk_zone is not None:
                weight *= 0.25 + ioa(point.bbox, desk_zone)
            score += weight
            if safe_zone is not None and ioa(point.bbox, safe_zone) >= config.cashier_safe_ioa:
                at_safe = True
        if at_safe:
            score *= config.cashier_safe_bonus
        inside_points = info.inside_points()
        if inside_points and inside_points[0].t <= 1.0:
            score *= config.cashier_start_bonus
        if score > best_score:
            best, best_score = info, score
    return best


def _workplace(cashier: _TrackInfo, desk: Optional[BBox], config: SceneConfig) -> Optional[BBox]:
    """Зона рабочего места: медиана рамок кассира у стола (или всех его рамок, если стол не найден)."""
    boxes = [p.bbox for p in cashier.inside_points()]
    if not boxes:
        return None
    if desk is not None:
        desk_zone = expand(desk, config.desk_margin)
        near = [b for b in boxes if ioa(b, desk_zone) >= config.workplace_min_desk_ioa]
        if near:
            boxes = near
    box = median_box(boxes)
    return expand(box, config.workplace_margin) if box else None


def _mark_extra_cashiers(infos: Sequence[_TrackInfo], cashier: _TrackInfo, workplace: Optional[BBox],
                         config: SceneConfig) -> None:
    """Трек, не пересекающийся по времени с кассиром (допуск cashier_overlap_sec: трекер начинает
    новый трек на месте кассира, пока старый ещё доживает) и сидящий на его месте, — тот же кассир."""
    if workplace is None:
        return
    main = cashier.track
    for info in infos:
        points = info.inside_points()
        if info.role == "cashier" or not points:
            continue
        tol = config.cashier_overlap_sec
        if not (points[-1].t < main.first_t + tol or points[0].t > main.last_t - tol):
            continue
        if median(ioa(p.bbox, workplace) for p in points) >= config.cashier_extra_ioa:
            info.role = "cashier"


def _near_door(box: BBox, door_zone: Optional[BBox], config: SceneConfig) -> bool:
    if door_zone is None:
        return False       # дверь не найдена: входов и выходов нет (предупреждение в meta.scene.warnings)
    cx, cy = center(box)
    return contains_point(door_zone, cx, cy) or ioa(box, door_zone) >= config.door_min_ioa


def _went_in(points: Sequence[TrackPoint], door_zone: BBox, config: SceneConfig) -> bool:
    """Человек действительно был в комнате, а не мелькнул в проёме (режим — SceneConfig.entry_mode)."""
    if points[-1].t - points[0].t + 1e-6 < config.min_room_sec:
        return False
    if config.entry_mode == "room":
        in_room = [p for p in points if not _near_door(p.bbox, door_zone, config)]
        return bool(in_room) and in_room[-1].t - in_room[0].t + 1e-6 >= config.min_room_sec
    if config.entry_mode == "move":
        for end in (points[0], points[-1]):
            others = points[1:] if end is points[0] else points[:-1]
            h = max(height(end.bbox), 1e-6)
            if any(center_dist(end.bbox, p.bbox) >= config.entry_move * h for p in others):
                return True
        return False
    return True


@dataclass(frozen=True)
class DoorEvent:
    """Вход или выход через дверь: позиция кадра, трек, кассир ли это."""

    pos: int
    track_id: int
    cashier: bool


def _door_events(infos: Sequence[_TrackInfo], door: Optional[BBox], duration: float,
                 config: SceneConfig) -> tuple[list[DoorEvent], list[DoorEvent]]:
    """Входы и выходы через дверь.

    Вход: первая точка трека внутри у двери, и человек пробыл внутри не меньше min_room_sec;
    режим entry_mode добавляет условие «сместился в комнату» (см. SceneConfig). Выход —
    последняя точка у двери при том же условии.
    """
    door_zone = pad(door, config.door_margin) if door else None
    entered: list[DoorEvent] = []
    exited: list[DoorEvent] = []
    if door_zone is None:
        return entered, exited
    for info in infos:
        points = info.inside_points()
        if not points or points[-1].t - points[0].t < config.min_event_track_sec:
            continue
        if not _went_in(points, door_zone, config):
            continue
        first, last = points[0], points[-1]
        cashier = info.role == "cashier"
        if first.t > config.entry_grace_sec and _near_door(first.bbox, door_zone, config):
            entered.append(DoorEvent(first.frame_index, info.track.track_id, cashier))
        if last.t < duration - config.exit_grace_sec and _near_door(last.bbox, door_zone, config):
            exited.append(DoorEvent(last.frame_index, info.track.track_id, cashier))
    return entered, exited



@dataclass(frozen=True)
class Departure:
    """Уход кассира с рабочего места: позиция кадра ухода, первого кадра вдали, возвращения (None — не вернулся)."""

    left_pos: int
    far_pos: int
    return_pos: Optional[int]


def _departures(boxes: Sequence[Optional[BBox]], scores: Sequence[float], times: Sequence[float],
                workplace: Optional[BBox], safe: Optional[BBox], config: SceneConfig) -> list[Departure]:
    """Уходы кассира с рабочего места по его рамке в каждом кадре (None — кассира нет в кадре).

    Расстояние до места — между центрами рамки и рабочего места в долях ширины рабочего места.
    Уход: кассир был у места (ближе leave_home), потом вдали (дальше leave_far, не у сейфа — сейф
    входит в рабочее место) и не возвращался не меньше leave_min_away_sec. Возвращение — кассир
    leave_return_sec подряд у места с score детекции не ниже leave_return_score: человек за
    стеклянной перегородкой в коридоре проходит мимо места с низким score и не засчитывается.
    """
    if workplace is None or not boxes:
        return []
    wx, wy = center(workplace)
    width = max(workplace[2] - workplace[0], 1e-6)
    safe_zone = expand(safe, config.desk_margin) if safe else None

    def dist(box: BBox) -> float:
        cx, cy = center(box)
        return math.hypot(cx - wx, cy - wy) / width

    home = [b is not None and dist(b) <= config.leave_home for b in boxes]
    steady = [h and sc >= config.leave_return_score for h, sc in zip(home, scores)]
    far = [b is not None and dist(b) >= config.leave_far
           and not (safe_zone is not None and ioa(b, safe_zone) >= config.cashier_safe_ioa) for b in boxes]
    seated = sorted(height(b) for b, h in zip(boxes, home) if h and b is not None)
    seated_h = seated[len(seated) // 5] if seated else 0.0

    def returned_at(start: int) -> Optional[int]:
        run = None
        for k in range(start, len(boxes)):
            if steady[k]:
                run = k if run is None else run
                if times[k] - times[run] + 1e-6 >= config.leave_return_sec:
                    return run
            else:
                run = None
        return None

    out: list[Departure] = []
    pos = 0
    while pos < len(boxes):
        far_pos = next((k for k in range(pos, len(boxes)) if far[k]), None)
        if far_pos is None:
            break
        last_home = next((k for k in range(far_pos - 1, -1, -1) if home[k]), None)
        back = returned_at(far_pos)
        end_t = times[back] if back is not None else times[-1]
        # не вернулся до конца ролика — уход, даже если ролик кончился вскоре после
        if last_home is None or (back is not None and end_t - times[far_pos] < config.leave_min_away_sec):
            pos = (back + 1) if back is not None else far_pos + 1
            continue
        left = last_home
        box = boxes[last_home]
        if seated_h > 0 and box is not None and height(box) >= config.leave_up_ratio * seated_h:
            k = last_home
            while k > 0 and times[last_home] - times[k - 1] <= config.leave_up_sec:
                prev = boxes[k - 1]
                if prev is None or height(prev) < config.leave_up_ratio * seated_h:
                    break
                k -= 1
            if k > 0 and boxes[k - 1] is not None and times[last_home] - times[k - 1] <= config.leave_up_sec:
                left = k
        out.append(Departure(left, far_pos, back))
        if back is None:
            break
        pos = back + 1
    return out


# --------------------------------------------------------------------------- предметы


def _max_near(dets: Sequence[Detection], labels: Sequence[str], zone: BBox, min_score: float) -> float:
    """Наибольший score предмета из labels, центр которого внутри zone (0, если таких нет)."""
    best = 0.0
    for det in dets:
        if det.label in labels and det.score >= min_score:
            cx, cy = center(det.bbox_xyxy)
            if contains_point(zone, cx, cy):
                best = max(best, det.score)
    return best


def _max_opt(*values: Optional[float]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _median3(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Медиана по окну из 3 точек; None остаётся None и не участвует в соседних окнах."""
    out: list[Optional[float]] = []
    for i, value in enumerate(values):
        if value is None:
            out.append(None)
            continue
        window = [v for v in values[max(0, i - 1): i + 2] if v is not None]
        out.append(median(window))
    return out


class _Held:
    """Значение zero-shot-оценки в момент t: последняя оценка не позже t (до первой — первая)."""

    def __init__(self, samples: Sequence[tuple[float, Optional[float]]]) -> None:
        ordered = sorted(samples, key=lambda s: s[0])
        self._times = [t for t, _ in ordered]
        self._values = [v for _, v in ordered]

    def at(self, t: float) -> Optional[float]:
        if not self._times:
            return None
        i = bisect_right(self._times, t + 1e-6) - 1
        return self._values[max(i, 0)]


# --------------------------------------------------------------------------- сборка


# Предметы на столе после ухода кассира (worker.scene.pipeline.check_left_items): запросы OWLv2.
# Последние четыре — опознаватели: рамка, больше похожая на них, — не ценность (телефон, машинка).
ITEM_QUERIES: tuple[str, ...] = (
    "a stack of banknotes", "paper money", "a bag", "a handbag", "a tablet computer", "a cup",
    "a laptop", "a telephone", "a money counting machine", "a printer",
)


# Группы предметов сканирования стола: поле состояния -> запросы OWLv2. worn — предмет, который бывает
# на человеке (одежда): рамка больше чем наполовину внутри рамки человека не считается.
DESK_ITEM_GROUPS: dict[str, tuple[str, ...]] = {
    "desk_cash": ("a stack of banknotes", "paper money"),
    "desk_cup": ("a cup",),
    "desk_coat": ("a coat", "a jacket"),
    "desk_collection_bag": ("a green cash collection bag",),
}
_WORN = {"desk_coat"}
_SCAN_DECOYS: tuple[str, ...] = ("a laptop", "a telephone", "a money counting machine", "a printer")
SCAN_QUERIES: tuple[str, ...] = tuple(dict.fromkeys(
    [q for qs in DESK_ITEM_GROUPS.values() for q in qs] + list(_SCAN_DECOYS)))


def with_desk_items(timeline: SceneTimeline, scan: Sequence[tuple[float, Sequence[Detection]]],
                    people: Optional[Mapping[float, Sequence[BBox]]] = None) -> SceneTimeline:
    """Поля desk_* по сканированию стола: в каждом состоянии — значение последнего скана не позже него.
    Предмет считается, если центр рамки в зоне стола (с запасом desk_margin), рамка не лежит почти
    целиком внутри более-менее уверенного опознавателя (ноутбук, телефон, машинка, принтер), а одежда —
    не на человеке (people: таймкод скана -> рамки людей)."""
    desk = timeline.zones.get("desk")
    cfg = timeline.meta.get("scene", {}).get("config", {})
    floor = float(cfg.get("item_scan_floor", 0.1))
    span = float(cfg.get("item_scan_span", 0.3))
    margin = float(cfg.get("desk_margin", 0.15))
    if desk is None or not scan:
        return timeline
    zone = expand(desk, margin)
    values: list[tuple[float, dict[str, float]]] = []
    for t, dets in sorted(scan, key=lambda x: x[0]):
        decoys = [d for d in dets if d.label in _SCAN_DECOYS]
        persons = (people or {}).get(round(t, 3), ())
        row: dict[str, float] = {}
        for field_name, queries in DESK_ITEM_GROUPS.items():
            best = 0.0
            for d in dets:
                if d.label not in queries:
                    continue
                cx, cy = center(d.bbox_xyxy)
                if not contains_point(zone, cx, cy):
                    continue
                if any(o.score >= 0.8 * d.score and ioa(d.bbox_xyxy, o.bbox_xyxy) >= 0.7 for o in decoys):
                    continue
                if field_name in _WORN and any(ioa(d.bbox_xyxy, pb) >= 0.5 for pb in persons):
                    continue
                best = max(best, d.score)
            row[field_name] = round(max(0.0, min(1.0, (best - floor) / span)), 3)
        values.append((t, row))
    times = [t for t, _ in values]
    states = []
    for s in timeline.states:
        i = bisect_right(times, s.t + 1e-6) - 1
        row = values[max(i, 0)][1]
        states.append(replace(s, extra={**s.extra, **row}))
    return replace(timeline, states=tuple(states))


def with_ft_items(timeline: SceneTimeline,
                  samples: Mapping[str, Sequence[tuple[float, Optional[float]]]]) -> SceneTimeline:
    """Поле desk_cash_ft — «наличные на столе» дообученной башни (SceneConfig.safe_ft, ряд
    state_samples.desk_cash раз в state_every_sec): в каждом состоянии — последняя оценка не позже него
    (как zero-shot-поля). Отдельное поле только для R14 (rules.yaml ft_field): desk_cash OWLv2-скана
    остаётся для всех прочих потребителей — башня путает банки напитка и планшет с деньгами (Х-24, Х-25).
    Нет ряда — сцена без изменений."""
    series = samples.get("desk_cash")
    if not series:
        return timeline
    held = _Held(series)
    states = [replace(s, extra={**s.extra, "desk_cash_ft": held.at(s.t)}) for s in timeline.states]
    return replace(timeline, states=tuple(states))


def people_at(scan: Sequence[tuple[float, Sequence[Detection]]], frames: Sequence[Any],
              tracks: Sequence[Track]) -> dict[float, list[BBox]]:
    """Рамки людей в кадрах сканирования стола: {таймкод скана: [рамки]} (frames — с timestamp_sec или t)."""
    stamps = [float(getattr(f, "timestamp_sec", getattr(f, "t", 0.0))) for f in frames]
    out: dict[float, list[BBox]] = {}
    for t, _ in scan:
        if not stamps:
            break
        pos = min(range(len(stamps)), key=lambda k: abs(stamps[k] - t))
        out[round(t, 3)] = [p.bbox for tr in tracks for p in tr.points if p.frame_index == pos]
    return out


def item_check_times(timeline: SceneTimeline) -> list[float]:
    """Таймкоды кадров для проверки стола после каждого ухода: через item_check_sec после того, как
    кассир оказался вдали (первое состояние cashier_away после cashier_left_workplace)."""
    cfg = timeline.meta.get("scene", {}).get("config", {})
    offsets = cfg.get("item_check_sec", SceneConfig().item_check_sec)
    states = timeline.states
    out: list[float] = []
    for i, s in enumerate(states):
        if not s.value("cashier_left_workplace"):
            continue
        far = next((k for k in range(i, len(states)) if states[k].value("cashier_away")), None)
        if far is None:
            continue
        for off in offsets:
            t = states[far].t + float(off)
            if t <= states[-1].t + 1e-6:
                out.append(round(t, 3))
    return sorted(set(out))


def with_left_items(timeline: SceneTimeline,
                    checks: Sequence[tuple[float, Sequence[Detection]]]) -> SceneTimeline:
    """Добавить к состоянию ухода (cashier_left_workplace) поле left_items — детекции OWLv2 на кадрах
    проверки стола до следующего ухода: [[таймкод, запрос, score, рамка]]."""
    states = list(timeline.states)
    lefts = [i for i, s in enumerate(states) if s.value("cashier_left_workplace")]
    for n, i in enumerate(lefts):
        until = states[lefts[n + 1]].t if n + 1 < len(lefts) else math.inf
        items = [[round(t, 3), d.label, round(float(d.score), 4), [round(v, 4) for v in d.bbox_xyxy]]
                 for t, dets in checks if states[i].t <= t < until for d in dets]
        states[i] = replace(states[i], extra={**states[i].extra, "left_items": items})
    return replace(timeline, states=tuple(states))


def with_brightness(timeline: SceneTimeline, values: Sequence[Optional[float]]) -> SceneTimeline:
    """Добавить в состояния поле brightness — средняя яркость кадра 0..1 (по одному значению на
    состояние). Детекторы не засчитывают переключение zero-shot-состояний на резком скачке света
    (Х-11, Х-12: свет меняется в момент ухода кассира)."""
    if len(values) != len(timeline.states):
        raise ValueError("brightness: ожидается по одному значению на состояние")
    states = tuple(
        replace(s, extra={**s.extra, "brightness": None if v is None else round(float(v), 3)})
        for s, v in zip(timeline.states, values)
    )
    return replace(timeline, states=states)


def with_door_change(timeline: SceneTimeline, diffs: Sequence[Optional[float]],
                     config: Optional[SceneConfig] = None) -> SceneTimeline:
    """Добавить в состояния door_diff — расхождение полосы двери с её видом в начале ролика
    (worker.perception.door.door_diffs, по одному значению на состояние) — и door_change: то же,
    приведённое к 0..1 порогами SceneConfig.door_diff_floor / door_diff_span (их можно менять без GPU
    по сырому кэшу). None — оценки нет (людей в полосе двери много, зоны двери нет)."""
    cfg = config or SceneConfig()
    if len(diffs) != len(timeline.states):
        raise ValueError("door_diff: ожидается по одному значению на состояние")
    span = max(cfg.door_diff_span, 1e-6)

    def change(d: Optional[float]) -> Optional[float]:
        return None if d is None else round(min(1.0, max(0.0, (float(d) - cfg.door_diff_floor) / span)), 3)

    states = tuple(
        replace(s, extra={**s.extra, "door_diff": None if d is None else round(float(d), 3), "door_change": change(d)})
        for s, d in zip(timeline.states, diffs)
    )
    return replace(timeline, states=states)


def assemble_timeline(
    *,
    video_id: str,
    duration_sec: float,
    frame_size: tuple[int, int],
    frames: Sequence[FrameRef],
    tracks: Sequence[Track],
    objects: Sequence[Sequence[Detection]],
    zones: Mapping[str, Optional[BBox]],
    state_samples: Mapping[str, Sequence[tuple[float, Optional[float]]]],
    config: Optional[SceneConfig] = None,
    meta: Optional[Mapping[str, Any]] = None,
    person_head: Optional[Sequence[Sequence[Any]]] = None,
    person_ft: Optional[Sequence[Sequence[Any]]] = None,
) -> SceneTimeline:
    """Собрать SceneTimeline.

    frames — проанализированные кадры по порядку; objects — предметы COCO (не люди) по одному
    списку на кадр; tracks — треки людей (frame_index = позиция кадра во frames); zones — стол,
    сейф, дверь, ноутбук (workplace считается здесь); state_samples — zero-shot-оценки
    {состояние: [(t, вероятность)]}, между оценками держится последнее значение.
    person_head — [(кадр, рамка, p)] головы «внутри кассы» по точкам исходных треков (SceneConfig.inside_head).
    person_ft — то же от дообученной башни (SceneConfig.inside_ft; без флага не используется).
    """
    cfg = config or SceneConfig()
    if len(objects) != len(frames):
        raise ValueError("objects: ожидается по одному списку детекций на кадр")
    times = [f.t for f in frames]
    duration = max(float(duration_sec), times[-1] if times else 0.0)
    desk, door = zones.get("desk"), zones.get("door")
    warnings: list[str] = []
    if desk is None:
        warnings.append("стол не найден: кассир выбран по размеру и длительности трека, предметы на столе = null")
    if door is None:
        warnings.append("дверь не найдена: входы и выходы не определяются")
    if zones.get("safe") is None:
        warnings.append("сейф не найден надёжно (worker.perception.zones.pick_safe): safe_open = null, R10 не проверяется")

    stitched = [t for t in stitch_tracks(tracks, cfg, door) if len(t.points) >= cfg.min_track_points]
    ref_height = _reference_height(stitched)
    head = {_head_key(fi, box): float(p) for fi, box, p in person_head} if person_head else None
    ft = {_head_key(fi, box): float(p) for fi, box, p in person_ft} if person_ft and cfg.inside_ft else None
    infos = [_TrackInfo(track=t, inside=_inside_flags(t, ref_height, desk, cfg, door, head, ft)) for t in stitched]
    for info in infos:
        _fill_frames(info, times, cfg)

    cashier = _select_cashier(infos, desk, cfg, zones.get("safe"))
    workplace: Optional[BBox] = None
    if cashier is None:
        warnings.append("кассир не определён: cashier_present = null")
    else:
        cashier.role = "cashier"
        workplace = _workplace(cashier, desk, cfg)
        _mark_extra_cashiers(infos, cashier, workplace, cfg)
    entered_ev, exited_ev = _door_events(infos, door, duration, cfg)
    entered = {e.pos for e in entered_ev}
    exited = {e.pos for e in exited_ev}

    def _ids(events: Sequence[DoorEvent], pos: int, cashier: bool) -> list[int]:
        return sorted(e.track_id for e in events if e.pos == pos and e.cashier == cashier)

    raw_presence: list[Optional[float]] = []
    per_frame: list[tuple[int, Optional[BBox], tuple[Person, ...]]] = []
    cashier_scores: list[float] = []
    for pos in range(len(frames)):
        inside = [(info, info.by_frame[pos]) for info in infos if pos in info.by_frame and info.by_frame[pos].inside]
        cashier_entries = [entry for info, entry in inside if info.role == "cashier"]
        cashier_bbox: Optional[BBox] = None
        cashier_score = 0.0
        presence: Optional[float]
        if cashier is None:
            presence = None
        elif cashier_entries:
            if workplace is not None:
                best = max(cashier_entries, key=lambda e: ioa(e.bbox, workplace))
                cashier_bbox, cashier_score = best.bbox, best.score
                presence = min(1.0, ioa(cashier_bbox, workplace) / cfg.presence_ioa)
            else:
                best = max(cashier_entries, key=lambda e: area(e.bbox))
                cashier_bbox, cashier_score = best.bbox, best.score
                presence = 1.0
        else:
            presence = 0.0
        raw_presence.append(presence)
        cashier_scores.append(cashier_score)
        others = tuple(
            Person(track_id=info.track.track_id, bbox=entry.bbox, first_seen_t=round(info.inside_points()[0].t, 3))
            for info, entry in inside if info.role != "cashier"
        )
        per_frame.append((len(inside), cashier_bbox, others))

    # Уход с рабочего места: от первого кадра вдали до возвращения кассира нет — рамка «кассира» в это
    # время чаще человек за стеклом или сам кассир в коридоре (worker.scene.builder._departures).
    departures = _departures([b for _, b, _ in per_frame], cashier_scores, times, workplace, zones.get("safe"), cfg)
    left_at = {d.left_pos for d in departures}
    away: set[int] = set()
    for d in departures:
        away.update(range(d.far_pos, d.return_pos if d.return_pos is not None else len(frames)))
    for pos in away:
        persons, cashier_bbox, others = per_frame[pos]
        if raw_presence[pos] is not None:
            raw_presence[pos] = 0.0
        per_frame[pos] = (persons - (1 if cashier_bbox is not None else 0), None, others)

    held = {name: _Held(samples) for name, samples in state_samples.items()}
    desk_zone = expand(desk, cfg.desk_margin) if desk else None
    states: list[SceneState] = []
    for pos, (ref, presence) in enumerate(zip(frames, _median3(raw_presence))):
        persons, cashier_bbox, others = per_frame[pos]
        dets = objects[pos]
        hand_zone = expand(cashier_bbox, cfg.hand_margin) if cashier_bbox else None
        zero_shot = {name: h.at(ref.t) for name, h in held.items()}
        states.append(SceneState(
            t=round(ref.t, 3),
            frame=ref.frame,
            persons_in_room=persons,
            cashier_present=None if presence is None else round(presence, 3),
            cashier_bbox=cashier_bbox,
            others=others,
            person_entered=pos in entered,
            person_exited=pos in exited,
            safe_open=zero_shot.get("safe_open"),
            door_open=zero_shot.get("door_open"),
            cash_on_desk=zero_shot.get("cash_on_desk"),
            bag_on_desk=_max_opt(
                _max_near(dets, BAG_LABELS, desk_zone, cfg.object_score) if desk_zone else None,
                zero_shot.get("bag_on_desk"),
            ),
            collection_bag=zero_shot.get("collection_bag"),
            outerwear_on_desk=zero_shot.get("outerwear_on_desk"),
            cup_or_food=_max_opt(
                _max_near(dets, FOOD_LABELS, hand_zone, cfg.object_score) if hand_zone else None,
                _max_near(dets, FOOD_LABELS, desk_zone, cfg.object_score) if desk_zone else None,
            ),
            phone_in_hand=_max_near(dets, PHONE_LABELS, hand_zone, cfg.object_score) if hand_zone else None,
            laptop_unlocked=zero_shot.get("laptop_unlocked"),
            # Новые поля (правило 7 контракта): входы и выходы посетителей (не кассира) и уход кассира.
            extra={
                "visitor_entered_ids": _ids(entered_ev, pos, False),
                "visitor_exited_ids": _ids(exited_ev, pos, False),
                "cashier_left_room": bool(_ids(exited_ev, pos, True)),
                "cashier_returned": bool(_ids(entered_ev, pos, True)),
                "cashier_left_workplace": pos in left_at,
                "cashier_away": pos in away,
            },
        ))

    out_zones: dict[str, Optional[BBox]] = {name: zones.get(name) for name in ZONE_NAMES}
    out_zones["workplace"] = workplace
    out_zones.update({name: box for name, box in zones.items() if name not in out_zones})
    full_meta = dict(meta or {})
    full_meta["scene"] = {
        "config": cfg.as_dict(),
        "reference_height": round(ref_height, 4),
        "tracks_raw": len(tracks),
        "tracks": len(stitched),
        "tracks_inside": sum(1 for info in infos if info.inside_points()),
        "cashier_track_ids": sorted(info.track.track_id for info in infos if info.role == "cashier"),
        "entered_t": [round(times[p], 3) for p in sorted(entered)],
        "exited_t": [round(times[p], 3) for p in sorted(exited)],
        "left_workplace_t": [round(times[d.left_pos], 3) for d in departures],
        "returned_t": [round(times[d.return_pos], 3) for d in departures if d.return_pos is not None],
        "warnings": warnings,
    }
    return SceneTimeline(
        video_id=video_id,
        duration_sec=round(duration, 3),
        fps_sampled=cfg.fps,
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        zones=out_zones,
        states=tuple(states),
        meta=full_meta,
    )
