#!/usr/bin/env python3
"""Метрика ухода кассира по кэшу сцен и эталонной разметке по кадрам (без GPU и моделей).

    PYTHONPATH=. python scripts/eval_departures.py --cache <кэш сцен> \\
        [--labels training/labels/cashier_departures.csv] [--tolerance 3.0] [--json out.json] \\
        [--raw <сырой кэш>] [--trace Х-26]

Разметка (training/labels/cashier_departures.csv) — одна строка на эпизод: kind = departure (ушёл с места),
not_departure (встал или отошёл, но не ушёл: сейф, наклон, заслонили), none (ролик просмотрен, ни ухода,
ни вставания). Времена — секунды (кадр / 3); «?» — разметчик не уверен, такое поле метрика не считает;
пусто — не было (не выходил, не вернулся до конца ролика).

Сцена: <cache>/<video_id>.json. Имена в разметке и в кэше сводятся без учёта Х/X. Из сцены берутся
cashier_left_workplace (уход), cashier_away (интервал ухода: конец — возвращение на место),
cashier_left_room (кассир вышел через дверь), cashier_returned (кассир вошёл через дверь), persons_in_room.
Время сцены сопоставляется по ключу в мс (int(round(t * 1000))).

Метрики:
  - уход: cashier_left_workplace против t_leave_workplace эпизодов departure. Найдено — событие в ±tolerance
    (один к одному, ближайшие первыми). Событие вне ±tolerance, но внутри эпизода ([t_stand − tol, t_return]),
    — «эпизод замечен, таймкод мимо»: для «найдено» это пропуск, в таймкоде — ошибка. Ложные — события,
    не отнесённые ни к одному эпизоду departure: на эпизоде not_departure, на ролике без ухода, прочие;
  - таймкод: по эпизодам, где модель отметила уход в пределах эпизода, — доля с |ошибкой| ≤ tolerance и
    медиана |ошибки|;
  - выход из кассы: cashier_left_room против t_exit_room — так же; возвращение: конец интервала cashier_away
    против t_return (и отдельно cashier_returned — вход кассира через дверь); «не вернулся до конца» тоже
    сверяется;
  - касса пустая: эпизоды departure с room_empty да/нет; модель говорит «пусто», если на интервале после
    ухода (с t_exit_room, иначе с t_leave_workplace, до t_return или конца) persons_in_room = 0 хотя бы в
    половине состояний;
  - смена трека: доля эпизодов departure с track_switch = да (колонка разметки).

--raw: сырой кэш тех же видео — для каждого эпизода ещё и id исходных треков RT-DETR под рамкой кассира
(рамка до вычитания ухода, как её видит builder._departures). --trace <видео>: по кадрам — рамка кассира,
score, расстояние до места в ширинах рабочего места, рост против сидячего, id трека (нужен --raw).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.scene.contract import SceneTimeline, load_scene  # noqa: E402

LABELS = ROOT / "training" / "labels" / "cashier_departures.csv"
KINDS = ("departure", "not_departure", "none")


def norm_id(video_id: str) -> str:
    """«Х-10» и «X-10» — одно видео."""
    return video_id.strip().replace("Х", "X").replace("х", "x")


def ms(t: float) -> int:
    return int(round(float(t) * 1000))


# --------------------------------------------------------------------------- разметка


def _time(value: str) -> Any:
    """'' -> None (не было), '?' -> '?' (не уверен), иначе секунды."""
    value = (value or "").strip().replace(",", ".")
    if not value:
        return None
    if value == "?":
        return "?"
    return float(value)


@dataclass
class Episode:
    video_id: str
    kind: str
    t_stand: Any = None
    t_leave: Any = None
    t_exit: Any = None
    t_return: Any = None
    room_empty: str = ""
    left_behind: str = ""
    track_switch: str = ""
    note: str = ""

    def known(self, name: str) -> Optional[float]:
        value = getattr(self, name)
        return value if isinstance(value, float) else None

    def start(self) -> Optional[float]:
        """Начало эпизода — встал (если известно), иначе ушёл с места."""
        return self.known("t_stand") if self.known("t_stand") is not None else self.known("t_leave")


def load_labels(path: Path) -> dict[str, list[Episode]]:
    """Эпизоды по видео (ключ — norm_id). Ролик с одной строкой kind=none — без эпизодов, но просмотрен."""
    out: dict[str, list[Episode]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            kind = row["kind"].strip()
            if kind not in KINDS:
                raise ValueError(f"{row['video_id']}: неизвестный kind {kind!r}")
            vid = row["video_id"].strip()
            episodes = out.setdefault(norm_id(vid), [])
            if kind == "none":
                continue
            episodes.append(Episode(
                video_id=vid, kind=kind, t_stand=_time(row["t_stand"]), t_leave=_time(row["t_leave_workplace"]),
                t_exit=_time(row["t_exit_room"]), t_return=_time(row["t_return"]),
                room_empty=row["room_empty"].strip(), left_behind=row["left_behind"].strip(),
                track_switch=row.get("track_switch", "").strip(), note=row.get("note", "").strip()))
    return out


# --------------------------------------------------------------------------- сцена


@dataclass
class Predicted:
    leave: list[float] = field(default_factory=list)          # cashier_left_workplace
    away: list[tuple[float, Optional[float]]] = field(default_factory=list)  # (первый кадр вдали, возвращение)
    exit: list[float] = field(default_factory=list)           # cashier_left_room
    door_return: list[float] = field(default_factory=list)    # cashier_returned
    empty_ms: dict[int, bool] = field(default_factory=dict)   # t (мс) -> persons_in_room == 0
    duration: float = 0.0


def predicted(scene: SceneTimeline) -> Predicted:
    p = Predicted(duration=float(scene.duration_sec))
    run_start: Optional[float] = None
    for s in scene.states:
        if s.value("cashier_left_workplace"):
            p.leave.append(float(s.t))
        if s.value("cashier_left_room"):
            p.exit.append(float(s.t))
        if s.value("cashier_returned"):
            p.door_return.append(float(s.t))
        away = bool(s.value("cashier_away"))
        if away and run_start is None:
            run_start = float(s.t)
        elif not away and run_start is not None:
            p.away.append((run_start, float(s.t)))
            run_start = None
        p.empty_ms[ms(s.t)] = s.persons_in_room == 0
    if run_start is not None:
        p.away.append((run_start, None))
    return p


def load_cache(cache: Path) -> dict[str, Path]:
    return {norm_id(f.stem): f for f in sorted(cache.glob("*.json")) if not f.name.startswith("_")}


# --------------------------------------------------------------------------- сопоставление


def match(truth: list[tuple[int, float]], pred: list[float], tolerance: float) -> dict[int, int]:
    """Один к одному в ±tolerance, ближайшие пары первыми: {индекс эталона: индекс предсказания}."""
    pairs = sorted((abs(p - t), i, j) for i, t in truth for j, p in enumerate(pred) if abs(p - t) <= tolerance + 1e-6)
    used_t: set[int] = set()
    used_p: set[int] = set()
    out: dict[int, int] = {}
    for _, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        out[i] = j
        used_t.add(i)
        used_p.add(j)
    return out


def _window(ep: Episode, duration: float, tolerance: float) -> tuple[float, float]:
    start = ep.start()
    start = (start if start is not None else 0.0) - tolerance
    end = ep.known("t_return")
    return start, (end if end is not None else duration) + tolerance


@dataclass
class Row:
    """Строка таблицы по эпизодам."""

    video_id: str
    kind: str
    field: str                  # leave | exit | return
    truth: Optional[float]
    pred: Optional[float]
    status: str                 # found | late | missed | false_* | ok_none | ...

    @property
    def error(self) -> Optional[float]:
        if self.truth is None or self.pred is None:
            return None
        return round(self.pred - self.truth, 2)


def _events(field_name: str, vid: str, episodes: list[Episode], pred: list[float], duration: float,
            tolerance: float, truth_attr: str) -> list[Row]:
    """Эпизоды departure с известным truth_attr против событий pred; остальные события — ложные."""
    deps = [(k, e) for k, e in enumerate(episodes) if e.kind == "departure"]
    truth = [(k, e.known(truth_attr)) for k, e in deps if e.known(truth_attr) is not None]
    strict = match(truth, pred, tolerance)
    rows: list[Row] = []
    used = set(strict.values())
    for k, t in truth:
        ep = episodes[k]
        if k in strict:
            rows.append(Row(vid, ep.kind, field_name, t, pred[strict[k]], "found"))
            continue
        lo, hi = _window(ep, duration, tolerance)
        inside = [j for j, p in enumerate(pred) if j not in used and lo <= p <= hi]
        if inside:
            j = min(inside, key=lambda j: abs(pred[j] - t))
            used.add(j)
            rows.append(Row(vid, ep.kind, field_name, t, pred[j], "late"))
        else:
            rows.append(Row(vid, ep.kind, field_name, t, None, "missed"))
    # события внутри эпизода departure, у которого это поле не размечено («?» или пусто), — не ложные
    for k, e in deps:
        if e.known(truth_attr) is not None:
            continue
        lo, hi = _window(e, duration, tolerance)
        for j, p in enumerate(pred):
            if j not in used and lo <= p <= hi:
                used.add(j)
                rows.append(Row(vid, e.kind, field_name, None, p, "unlabeled"))
    has_departure = bool(deps)
    for j, p in enumerate(pred):
        if j in used:
            continue
        nd = [e for e in episodes if e.kind == "not_departure"
              and _window(e, duration, tolerance)[0] <= p <= _window(e, duration, tolerance)[1]]
        if nd:
            status = "false_not_departure"
        elif not has_departure:
            status = "false_no_departure"
        else:
            status = "false_other"
        rows.append(Row(vid, nd[0].kind if nd else "", field_name, None, p, status))
    return rows


def _returns(vid: str, episodes: list[Episode], pred: Predicted, tolerance: float) -> list[Row]:
    """Возвращение на место: конец интервала cashier_away, сопоставленного эпизоду по началу (±tol + эпизод)."""
    rows: list[Row] = []
    starts = [a for a, _ in pred.away]
    for e in episodes:
        if e.kind != "departure" or e.t_return == "?":
            continue
        lo, hi = _window(e, pred.duration, tolerance)
        cand = [i for i, a in enumerate(starts) if lo <= a <= hi]
        truth = e.known("t_return")
        if not cand:
            rows.append(Row(vid, e.kind, "return", truth, None, "no_departure"))
            continue
        back = pred.away[cand[0]][1]
        if truth is None:
            rows.append(Row(vid, e.kind, "return", None, back, "ok_none" if back is None else "false_return"))
        elif back is None:
            rows.append(Row(vid, e.kind, "return", truth, None, "missed"))
        else:
            rows.append(Row(vid, e.kind, "return", truth, back, "found" if abs(back - truth) <= tolerance + 1e-6
                            else "late"))
    return rows


def _room_empty(vid: str, episodes: list[Episode], pred: Predicted) -> list[dict]:
    out = []
    for e in episodes:
        if e.kind != "departure" or e.room_empty not in ("да", "нет"):
            continue
        start = e.known("t_exit") if e.known("t_exit") is not None else e.known("t_leave")
        if start is None:
            continue
        end = e.known("t_return") if e.known("t_return") is not None else pred.duration
        flags = [v for k, v in pred.empty_ms.items() if ms(start) <= k <= ms(end)]
        if not flags:
            continue
        share = sum(flags) / len(flags)
        out.append({"video_id": vid, "truth": e.room_empty, "pred_empty_share": round(share, 2),
                    "pred": "да" if share >= 0.5 else "нет", "ok": (share >= 0.5) == (e.room_empty == "да")})
    return out


# --------------------------------------------------------------------------- треки (--raw)


def cashier_trace(raw_path: Path) -> list[dict]:
    """По кадрам: рамка кассира до вычитания ухода (вход builder._departures), score, расстояние до места
    в ширинах рабочего места, рост против сидячего, id исходного трека RT-DETR (рамка — копия его точки)."""
    import worker.scene.builder as builder
    from worker.scene.raw_cache import load_raw, rebuild_scene

    sys.path.insert(0, str(ROOT / "scripts"))
    from rebuild_from_raw import PERCEPTION_FIELDS

    raw = load_raw(raw_path)
    keep = {k: raw["config"][k] for k in PERCEPTION_FIELDS if k in raw.get("config", {})}
    config = dataclasses.replace(builder.SceneConfig(), **keep)
    seen: dict[str, Any] = {}
    original = builder._departures

    def spy(boxes, scores, times, workplace, safe, cfg):
        seen.update(boxes=list(boxes), scores=list(scores), times=list(times), workplace=workplace, safe=safe)
        return original(boxes, scores, times, workplace, safe, cfg)

    builder._departures = spy
    try:
        rebuild_scene(raw, config)
    finally:
        builder._departures = original
    owner: dict[tuple, int] = {}
    for tr in raw["tracks"]:
        for _, _, box, _ in tr["points"]:
            owner.setdefault(tuple(round(x, 4) for x in box), int(tr["id"]))
    wp = seen.get("workplace")
    out = []
    for box, score, t in zip(seen.get("boxes", []), seen.get("scores", []), seen.get("times", [])):
        row: dict[str, Any] = {"t": round(t, 3), "track": None, "score": round(score, 3), "dist": None, "h": None,
                               "box": None if box is None else [round(x, 3) for x in box]}
        if box is not None:
            row["track"] = owner.get(tuple(round(x, 4) for x in box))
            row["h"] = round(box[3] - box[1], 3)
            if wp is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                wx, wy = (wp[0] + wp[2]) / 2, (wp[1] + wp[3]) / 2
                row["dist"] = round(math.hypot(cx - wx, cy - wy) / max(wp[2] - wp[0], 1e-6), 2)
        out.append(row)
    return out


def tracks_in(trace: list[dict], lo: float, hi: float) -> list[int]:
    """Id треков под рамкой кассира на [lo, hi] в порядке появления."""
    ids: list[int] = []
    for row in trace:
        if lo - 1e-6 <= row["t"] <= hi + 1e-6 and row["track"] is not None and row["track"] not in ids:
            ids.append(row["track"])
    return ids


# --------------------------------------------------------------------------- сводка


def _summary(rows: list[Row], field_name: str, tolerance: float) -> dict:
    sel = [r for r in rows if r.field == field_name]
    found = sum(r.status == "found" for r in sel)
    late = sum(r.status == "late" for r in sel)
    missed = sum(r.status == "missed" for r in sel)
    errs = [abs(r.error) for r in sel if r.status in ("found", "late") and r.error is not None]
    out = {
        "truth": found + late + missed,
        "found": found,
        "late": late,
        "missed": missed,
        "false": sum(r.status.startswith("false") for r in sel),
        "false_not_departure": sum(r.status == "false_not_departure" for r in sel),
        "false_no_departure": sum(r.status == "false_no_departure" for r in sel),
        "false_other": sum(r.status == "false_other" for r in sel),
        "timecode_ok": sum(e <= tolerance + 1e-6 for e in errs),
        "timecode_n": len(errs),
        "median_abs_error": round(median(errs), 2) if errs else None,
    }
    return out


def evaluate(cache: Path, labels: dict[str, list[Episode]], tolerance: float,
             raw: Optional[Path] = None) -> dict:
    scenes = load_cache(cache)
    rows: list[Row] = []
    empty: list[dict] = []
    per_video: list[dict] = []
    missing = sorted(v for v in labels if v not in scenes)
    raw_files = load_cache(raw) if raw else {}
    for vid in sorted(labels):
        if vid not in scenes:
            continue
        episodes = labels[vid]
        pred = predicted(load_scene(scenes[vid]))
        vrows = (_events("leave", vid, episodes, pred.leave, pred.duration, tolerance, "t_leave")
                 + _events("exit", vid, episodes, pred.exit, pred.duration, tolerance, "t_exit")
                 + _returns(vid, episodes, pred, tolerance))
        door = _events("door_return", vid, episodes, pred.door_return, pred.duration, tolerance, "t_return")
        rows.extend(vrows + door)
        empty.extend(_room_empty(vid, episodes, pred))
        trace = cashier_trace(raw_files[vid]) if vid in raw_files else None
        eps = []
        for e in episodes:
            item = {k: v for k, v in asdict(e).items() if k != "note"}
            if trace is not None:
                lo, hi = _window(e, pred.duration, 0.0)
                item["model_tracks"] = tracks_in(trace, lo, hi)
            eps.append(item)
        per_video.append({"video_id": vid, "episodes": eps, "pred_leave": pred.leave, "pred_away": pred.away,
                          "pred_exit": pred.exit, "pred_door_return": pred.door_return,
                          "rows": [{**asdict(r), "error": r.error} for r in vrows + door]})
    deps = [e for v in labels if v in scenes for e in labels[v] if e.kind == "departure"]
    switch_known = [e for e in deps if e.track_switch in ("да", "нет")]
    summary = {
        "videos": sum(1 for v in labels if v in scenes),
        "videos_missing_in_cache": missing,
        "videos_unlabeled": sorted(v for v in scenes if v not in labels),
        "episodes": {k: sum(e.kind == k for v in labels if v in scenes for e in labels[v]) for k in KINDS[:2]},
        "tolerance": tolerance,
        "leave": _summary(rows, "leave", tolerance),
        "exit": _summary(rows, "exit", tolerance),
        "return": {**_summary(rows, "return", tolerance),
                   "no_return_ok": sum(r.status == "ok_none" for r in rows if r.field == "return"),
                   "false_return": sum(r.status == "false_return" for r in rows if r.field == "return"),
                   "no_departure": sum(r.status == "no_departure" for r in rows if r.field == "return")},
        "door_return": _summary(rows, "door_return", tolerance),
        "room_empty": {"n": len(empty), "ok": sum(x["ok"] for x in empty)},
        "track_switch": {"n": len(switch_known), "yes": sum(e.track_switch == "да" for e in switch_known)},
    }
    return {"summary": summary, "room_empty": empty, "videos": per_video}


def _fmt(t: Optional[float]) -> str:
    return "—" if t is None else f"{t:.1f}"


def print_report(res: dict) -> None:
    s = res["summary"]
    tol = s["tolerance"]
    print(f"Видео: {s['videos']}, эпизодов departure {s['episodes']['departure']}, "
          f"not_departure {s['episodes']['not_departure']}; совпадение ±{tol:g} с")
    if s["videos_missing_in_cache"]:
        print("нет в кэше:", ", ".join(s["videos_missing_in_cache"]))
    if s["videos_unlabeled"]:
        print("не размечены:", ", ".join(s["videos_unlabeled"]))
    names = {"leave": "Уход (cashier_left_workplace)", "exit": "Выход из кассы (cashier_left_room)",
             "return": "Возвращение на место (конец cashier_away)",
             "door_return": "Вход кассира через дверь (cashier_returned) против t_return"}
    for key, title in names.items():
        m = s[key]
        line = (f"{title}: эталон {m['truth']}, найдено {m['found']}, таймкод мимо {m['late']}, пропущено "
                f"{m['missed']}, ложных {m['false']} (на not_departure {m['false_not_departure']}, "
                f"на роликах без ухода {m['false_no_departure']}, прочих {m['false_other']})")
        print(line)
        if m["timecode_n"]:
            print(f"  таймкод: {m['timecode_ok']} из {m['timecode_n']} в ±{tol:g} с, медиана |ошибки| "
                  f"{m['median_abs_error']} с")
    r = s["return"]
    print(f"  «не вернулся до конца»: верно {r['no_return_ok']}, ложное возвращение {r['false_return']}, "
          f"уход не найден {r['no_departure']}")
    e = s["room_empty"]
    print(f"Касса пустая после ухода: верно {e['ok']} из {e['n']}")
    ts = s["track_switch"]
    print(f"Смена трека кассира в эпизоде: {ts['yes']} из {ts['n']}")
    print()
    print(f"{'видео':<8} {'эпизод':<14} {'поле':<7} {'истина':>7} {'модель':>7} {'ошибка':>7}  статус")
    for v in res["videos"]:
        for r in v["rows"]:
            if r["field"] == "door_return":
                continue
            err = "—" if r["error"] is None else f"{r['error']:+.1f}"
            print(f"{v['video_id']:<8} {r['kind'] or '—':<14} {r['field']:<7} {_fmt(r['truth']):>7} "
                  f"{_fmt(r['pred']):>7} {err:>7}  {r['status']}")


def print_trace(trace: list[dict]) -> None:
    print(f"{'t':>6} {'трек':>5} {'score':>6} {'dist':>5} {'h':>6}  рамка")
    for row in trace:
        print(f"{row['t']:>6.2f} {str(row['track']):>5} {row['score']:>6.2f} {str(row['dist']):>5} "
              f"{str(row['h']):>6}  {row['box']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Метрика ухода кассира по кэшу сцен")
    ap.add_argument("--cache", type=Path, required=True, help="кэш сцен (<video_id>.json)")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--raw", type=Path, default=None, help="сырой кэш: id треков под рамкой кассира")
    ap.add_argument("--trace", default=None, help="видео: рамка кассира по кадрам (нужен --raw)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if args.trace:
        if args.raw is None:
            ap.error("--trace требует --raw")
        files = load_cache(args.raw)
        if norm_id(args.trace) not in files:
            print(f"{args.trace}: нет в {args.raw}", file=sys.stderr)
            return 2
        print_trace(cashier_trace(files[norm_id(args.trace)]))
        return 0
    if not load_cache(args.cache):
        print(f"в {args.cache} нет сцен *.json", file=sys.stderr)
        return 2
    res = evaluate(args.cache, load_labels(args.labels), args.tolerance, args.raw)
    print_report(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
