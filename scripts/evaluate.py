"""Офлайн-оценка качества на размеченном датасете (критерии 1–2 ТЗ).

Запуск из корня репозитория:
    PYTHONPATH=. python scripts/evaluate.py [--dataset dataset] [--annotations dataset/annotations.csv]
        [--rules rules.yaml] [--tolerance 3.0] [--limit N] [--json out.json]

Каждое видео (*.mov/*.MOV рекурсивно в --dataset) прогоняется через worker.runner.analyze напрямую:
без HTTP и без БД, рабочий каталог — временный.
Эталон: annotations.csv (video_id,t_label_sec,what_visible,proposed_type,rule_id,confidence);
video_id — как в open_labels.xlsx («X-10»; Х/X и суффикс « (1)» нормализуются). Видео в --dataset
опознаётся по md5 содержимого (KNOWN_VIDEOS), а не по имени файла; неизвестный файл — по имени.
Строка с непустым t_label_sec = одно событие-нарушение;
видео без строк или только с пустыми t_label_sec = «нет нарушения».

Метрики:
  - вердикт на всё видео: TP/FN/FP/TN, precision, recall (видео с ошибкой анализа в матрицу не входят);
  - таймкод: доля эталонных событий, для которых есть предсказанный timecode_sec в пределах
    ±tolerance с, и средняя |ошибка| по совпавшим (берётся ближайший предсказанный таймкод);
  - время обработки на видео и число видео дольше 300 с (лимит ТЗ).
Код возврата: 0; 2 — не найдено ни одного видео; 1 — некорректный rules.yaml.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from worker.runner import analyze, load_rules

TIME_LIMIT_SEC = 300.0
VIDEO_SUFFIX = ".mov"


@dataclass
class VideoReport:
    video_id: str
    path: str
    truth_violation: bool
    truth_timecodes: list[float]
    annotated: bool
    pred_violation: bool | None = None
    pred_timecodes: list[float] = field(default_factory=list)
    matched: int = 0
    abs_errors: list[float] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error: str | None = None

    @property
    def outcome(self) -> str:
        if self.pred_violation is None:
            return "ERR"
        if self.truth_violation:
            return "TP" if self.pred_violation else "FN"
        return "FP" if self.pred_violation else "TN"


def warn(msg: str) -> None:
    print(f"ПРЕДУПРЕЖДЕНИЕ: {msg}", file=sys.stderr)


_COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")


def normalize_video_id(raw: str) -> str:
    """«Х-10 (1)» -> «X-10»: в датасете кириллическая Х вперемешку с латинской X,
    а у скачанных копий суффикс « (1)»."""
    return _COPY_SUFFIX.sub("", raw.strip()).replace("Х", "X").replace("х", "x")


def load_annotations(path: Path) -> tuple[dict[str, list[float]], bool]:
    """video_id -> список эталонных таймкодов (пустой = нет нарушения). Второе значение — найден ли файл."""
    truth: dict[str, list[float]] = {}
    if not path.is_file():
        warn(f"файл разметки не найден: {path} — все видео считаются «без нарушения»")
        return truth, False
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"video_id", "t_label_sec"} - set(reader.fieldnames or [])
        if missing:
            warn(f"{path}: нет колонок {sorted(missing)} — разметка не используется")
            return truth, False
        for lineno, row in enumerate(reader, start=2):
            vid = normalize_video_id(row.get("video_id") or "")
            if not vid:
                continue
            events = truth.setdefault(vid, [])
            raw = (row.get("t_label_sec") or "").strip()
            if not raw:
                continue
            try:
                events.append(float(raw.replace(",", ".")))
            except ValueError:
                warn(f"{path}:{lineno}: некорректный t_label_sec={raw!r} — строка пропущена")
    return truth, True


# md5 файла -> video_id из open_labels.xlsx (имена файлов в open.zip организаторов).
# Видео опознаётся по содержимому, а не по имени: локальные копии в dataset/videos
# переименованы со сдвигом («Х-10 (1).mov» содержит X-8 и т.д.), датасет не трогаем.
# Несколько id у одного хэша — побайтно одинаковые ролики организаторов; эталон для них —
# объединение таймкодов всех id.
KNOWN_VIDEOS: dict[str, tuple[str, ...]] = {
    "767f2a65c96f8253bbe8e2f9d700e2a3": ("X-1",),
    "cdf8c6d946765256427f68ffa89974f1": ("X-2",),
    "c5efdc077357191503d132c9b0ccca1d": ("X-3",),
    "fdd5aa51aefe4b42dd4d318bbbd3f207": ("X-4", "X-47"),
    "304e8be8155a1f6529fe86ba88d05668": ("X-5", "X-48"),
    "dacb4cf5da210be8550081264ca31317": ("X-8",),
    "cc15799ee16a1c9d15fdbd7531d046bf": ("X-10",),
    "52d8217b15bc14b128e5d36d675bb98a": ("X-11",),
    "5c78378409c487c886b479d6fe561d0d": ("X-12",),
    "1fb32d10e1499480865e6b72199c6a3d": ("X-13",),
    "91f6d6e30c28c1d899ce874a06444561": ("X-14",),
    "d4e7c11733f8a6982fce711cf62c6a90": ("X-15",),
    "d4d0dce8ff94ba59eb6be0780b18339e": ("X-16",),
    "1cff94bb96d75439b00365a07adee716": ("X-17",),
    "abebea33f273d52bd3f50fb57c5444b9": ("X-18",),
    "595c577e53299940a5c557bf84c795a2": ("X-19",),
    "c19c3785b4a311ee97a6d0df1e9bddd6": ("X-20",),
    "7356dd59916c98eae91b2710b15ccb27": ("X-21",),
    "b88d9942288015ffe424d5a50f206ab2": ("X-22",),
    "cc739e64be12fc0acdeb582d4fdc0463": ("X-23",),
    "b18427a9a72818c2768fbf501ec2d9e3": ("X-24",),
    "8661055b1e0d0d2b083347eff3b581f7": ("X-25",),
    "23bc05eb011f80545a6009d98836baa6": ("X-26",),
    "9183adf045efbbc3cb115605e9ca704b": ("X-27",),
    "545c045acd070acb09ff2594f1b44577": ("X-28",),
    "4c92ba52ec957d0ecf81a09f66817b29": ("X-29",),
    "0608eca654509fddf4bb0dc799e2e6de": ("X-30",),
    "442dc88203d3715c5293e7da2156bd35": ("X-31",),
    "e00ced6c40089f399ab909d11629a5df": ("X-32",),
    "5575e411e2810a72703890f0b2443c10": ("X-33",),
    "a91d22affe251b634dc0d7fd63235c9e": ("X-34",),
    "d19984330918f5f57453950acce2118a": ("X-35",),
    "189e0353de852ff8f2ab3cd977db2a0f": ("X-37",),
    "e5a64fff53efade025ae10e6bbd047df": ("X-38",),
    "5fd1b371c742b676b6c029d16c2e2198": ("X-39",),
    "612cf89dc4e7213143242bc409949ded": ("X-40",),
    "d14976d777c2ecddbfd6f31f2359f4e0": ("X-43",),
    "c41b35b2ac643cf436b0f6d548d5c163": ("X-44",),
    "3ba888f705c81cad73bc45ece2077ff4": ("X-45",),
    "e27b34f262cc51c90b44deda61422f56": ("X-46",),
}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_videos(dataset: Path) -> tuple[dict[str, Path], dict[str, tuple[str, ...]]]:
    """video_id -> файл и video_id -> все id разметки с тем же содержимым.

    id берётся из KNOWN_VIDEOS по md5; неизвестный файл — по нормализованному имени.
    Файл с уже встреченным содержимым пропускается (дубль)."""
    videos: dict[str, Path] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    if not dataset.is_dir():
        warn(f"каталог датасета не найден: {dataset}")
        return videos, aliases
    for p in sorted(dataset.rglob("*")):
        if not (p.is_file() and p.suffix.lower() == VIDEO_SUFFIX):
            continue
        by_name = normalize_video_id(p.stem)
        ids = KNOWN_VIDEOS.get(file_md5(p), (by_name,))
        vid = ids[0]
        if vid in videos:
            warn(f"{p.name}: то же содержимое, что у {videos[vid].name} ({vid}) — пропущен")
            continue
        if by_name not in ids:
            warn(f"{p.name}: по содержимому это {'/'.join(ids)}, а не {by_name}")
        videos[vid] = p
        aliases[vid] = ids
    return videos, aliases


def evaluate_one(rep: VideoReport, video: Path, rules: list, tmp_root: Path, tolerance: float) -> None:
    work_dir = tmp_root / rep.video_id
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        result = analyze(video, rep.video_id, work_dir, rules)
    except Exception as exc:  # noqa: BLE001 — ошибка одного видео не останавливает прогон
        rep.elapsed_sec = time.monotonic() - t0
        rep.error = f"{type(exc).__name__}: {exc}"
        return
    rep.elapsed_sec = time.monotonic() - t0
    if result.status.value != "done" or result.verdict is None:
        rep.error = f"status={result.status.value}"
        return
    rep.pred_violation = result.verdict.value == "violation"
    rep.pred_timecodes = [v.timecode_sec for v in result.violations]
    for t in rep.truth_timecodes:
        if not rep.pred_timecodes:
            break
        err = min(abs(p - t) for p in rep.pred_timecodes)
        if err <= tolerance:
            rep.matched += 1
            rep.abs_errors.append(err)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def summarize(reports: list[VideoReport], tolerance: float) -> dict:
    counts = {k: sum(1 for r in reports if r.outcome == k) for k in ("TP", "FN", "FP", "TN", "ERR")}
    tp, fn, fp = counts["TP"], counts["FN"], counts["FP"]
    truth_events = sum(len(r.truth_timecodes) for r in reports)
    matched = sum(r.matched for r in reports)
    errors = [e for r in reports for e in r.abs_errors]
    times = [r.elapsed_sec for r in reports]
    return {
        "videos": len(reports),
        **counts,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "tolerance_sec": tolerance,
        "truth_events": truth_events,
        "matched_events": matched,
        "timecode_hit_rate": _ratio(matched, truth_events),
        "timecode_mae_sec": (sum(errors) / len(errors)) if errors else None,
        "time_mean_sec": (sum(times) / len(times)) if times else None,
        "time_max_sec": max(times) if times else None,
        "over_time_limit": sum(1 for t in times if t > TIME_LIMIT_SEC),
    }


def print_table(reports: list[VideoReport]) -> None:
    head = f"{'video_id':<28} {'эталон':<8} {'прогноз':<8} {'итог':<4} {'эталон t, с':<18} {'прогноз t, с':<18} {'совп.':<6} {'время, с':>8}"
    print(head)
    print("-" * len(head))
    for r in reports:
        truth = "наруш." if r.truth_violation else "нет"
        pred = "—" if r.pred_violation is None else ("наруш." if r.pred_violation else "нет")
        tt = ",".join(f"{t:g}" for t in r.truth_timecodes) or "—"
        pt = ",".join(f"{t:g}" for t in r.pred_timecodes) or "—"
        hit = f"{r.matched}/{len(r.truth_timecodes)}" if r.truth_timecodes else "—"
        print(f"{r.video_id[:28]:<28} {truth:<8} {pred:<8} {r.outcome:<4} {tt[:18]:<18} {pt[:18]:<18} {hit:<6} {r.elapsed_sec:>8.2f}")
        if r.error:
            print(f"    ошибка: {r.error}")


def print_summary(s: dict) -> None:
    print()
    print(f"Видео: {s['videos']}  (ошибок анализа: {s['ERR']}, в матрицу не входят)")
    print(f"Вердикт по видео: TP={s['TP']} FN={s['FN']} FP={s['FP']} TN={s['TN']}  "
          f"precision={_fmt(s['precision'])} recall={_fmt(s['recall'])}")
    print(f"Таймкод (±{s['tolerance_sec']:g} с): совпало {s['matched_events']}/{s['truth_events']} "
          f"эталонных событий = {_fmt(s['timecode_hit_rate'], pct=True)}; "
          f"средняя |ошибка| по совпавшим = {_fmt(s['timecode_mae_sec'])} с")
    print(f"Время обработки: среднее {_fmt(s['time_mean_sec'])} с, максимум {_fmt(s['time_max_sec'])} с, "
          f"дольше {TIME_LIMIT_SEC:g} с: {s['over_time_limit']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка worker.runner.analyze на размеченном датасете")
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--annotations", type=Path, default=None,
                    help="CSV разметки (по умолчанию dataset/annotations.csv)")
    ap.add_argument("--rules", type=Path, default=Path("rules.yaml"))
    ap.add_argument("--tolerance", type=float, default=3.0, help="допуск таймкода, с (по умолчанию 3.0)")
    ap.add_argument("--limit", type=int, default=None, help="обработать не больше N видео")
    ap.add_argument("--json", type=Path, default=None, help="сохранить отчёт в JSON")
    args = ap.parse_args(argv)
    if args.annotations is None:
        args.annotations = Path("dataset") / "annotations.csv"
    if args.tolerance < 0:
        ap.error("--tolerance должен быть >= 0")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit должен быть > 0")

    try:
        rules = load_rules(args.rules)
    except ValueError as exc:
        print(f"rules не загружены: {exc}", file=sys.stderr)
        return 1
    truth, _ = load_annotations(args.annotations)
    videos, aliases = find_videos(args.dataset)
    covered = {a for ids in aliases.values() for a in ids}

    for vid in sorted(set(truth) - covered):
        warn(f"видео {vid!r} есть в разметке, но не найдено в {args.dataset}")
    for vid in sorted(v for v in videos if not set(aliases[v]) & set(truth)):
        warn(f"видео {vid!r} без разметки — считается «без нарушения»")
    if not videos:
        print(f"Не найдено ни одного видео *{VIDEO_SUFFIX} в {args.dataset}", file=sys.stderr)
        return 2

    items = list(videos.items())
    if args.limit is not None:
        items = items[: args.limit]

    reports: list[VideoReport] = []
    with tempfile.TemporaryDirectory(prefix="evaluate-") as tmp:
        for i, (vid, path) in enumerate(items, start=1):
            tcs = sorted({t for a in aliases[vid] for t in truth.get(a, [])})
            rep = VideoReport(video_id="+".join(aliases[vid]), path=str(path), truth_violation=bool(tcs),
                              truth_timecodes=tcs, annotated=any(a in truth for a in aliases[vid]))
            print(f"[{i}/{len(items)}] {vid} ...", file=sys.stderr, flush=True)
            evaluate_one(rep, path, rules, Path(tmp), args.tolerance)
            reports.append(rep)

    print_table(reports)
    summary = summarize(reports, args.tolerance)
    print_summary(summary)

    if args.json is not None:
        payload = {
            "summary": summary,
            "videos": [{**asdict(r), "outcome": r.outcome} for r in reports],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON-отчёт: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
