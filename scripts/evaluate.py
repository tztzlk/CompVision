"""Офлайн-оценка качества на размеченном датасете (критерии 1–2 ТЗ).

Запуск из корня репозитория:
    PYTHONPATH=. python scripts/evaluate.py [--dataset dataset] [--annotations dataset/annotations.csv]
        [--rules rules.yaml] [--tolerance 3.0] [--limit N] [--json out.json]

Каждое видео (*.mov/*.MOV рекурсивно в --dataset) прогоняется через worker.runner.analyze напрямую:
без HTTP и без БД, рабочий каталог — временный.
Эталон: annotations.csv (video_id,t_label_sec,what_visible,proposed_type,rule_id,confidence);
video_id = имя файла без расширения; строка с непустым t_label_sec = одно событие-нарушение;
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
import json
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
            vid = (row.get("video_id") or "").strip()
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


def find_videos(dataset: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    if not dataset.is_dir():
        warn(f"каталог датасета не найден: {dataset}")
        return videos
    for p in sorted(dataset.rglob("*")):
        if p.is_file() and p.suffix.lower() == VIDEO_SUFFIX:
            if p.stem in videos:
                warn(f"повторяющийся video_id {p.stem!r}: {p} пропущен (взят {videos[p.stem]})")
                continue
            videos[p.stem] = p
    return videos


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
    videos = find_videos(args.dataset)

    for vid in sorted(set(truth) - set(videos)):
        warn(f"видео {vid!r} есть в разметке, но не найдено в {args.dataset}")
    for vid in sorted(set(videos) - set(truth)):
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
            tcs = truth.get(vid, [])
            rep = VideoReport(video_id=vid, path=str(path), truth_violation=bool(tcs),
                              truth_timecodes=tcs, annotated=vid in truth)
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
