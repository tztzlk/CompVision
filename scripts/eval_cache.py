#!/usr/bin/env python3
"""Замер правил по готовому кэшу сцен (без GPU, без моделей, без pydantic) — около минуты на 40 видео.

    PYTHONPATH=. python scripts/eval_cache.py [--cache data/scene_cache] [--annotations dataset/annotations.csv]
        [--rules rules.yaml] [--tolerance 3.0] [--json out.json] [--csv per_video.csv]

Сцена берётся из <cache>/<video_id>.json (video_id = имя файла видео, как пишет build_scene_cache.py).
Видео опознаётся по имени через KNOWN_VIDEOS из scripts/evaluate.py: дубликаты (Х-4 ≡ Х-47,
Х-5 ≡ Х-48) считаются одним видео, эталон — объединение их таймкодов. Итог — по уникальным видео.

Вердикт считается так же, как в ответе API: события выше min_confidence своего правила, не больше
MAX_VIOLATIONS (worker.events.selection). Кадры-доказательства не вырезаются: в сервисе кадр есть
у каждого события (кадры анализа лежат в work_dir/frames), поэтому вердикт совпадает.

Метрики:
  - вердикт по видео: TP/FN/FP/TN, precision, recall, доля верных вердиктов;
  - таймкод: доля эталонных событий, рядом с которыми (±tolerance) есть событие любого типа;
  - тип: строки разметки с rule_id, рядом с которыми есть событие ИМЕННО этого правила;
  - по правилам: на скольких чистых видео срабатывает (ложные тревоги), на скольких видео с
    нарушением, сколько эталонных событий накрывает в ±tolerance, сколько событий в первые 1,5 с.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.events import detect_events, load_rules  # noqa: E402
from worker.events.selection import top_events  # noqa: E402
from worker.scene.contract import load_scene  # noqa: E402

EARLY_SEC = 1.5


def normalize_video_id(raw: str) -> str:
    """«Х-10 (1)» -> «X-10» (как scripts/evaluate.normalize_video_id, без импорта worker.runner)."""
    import re

    return re.sub(r"\s*\(\d+\)$", "", raw.strip()).replace("Х", "X").replace("х", "x")


def known_aliases() -> dict[str, tuple[str, ...]]:
    """video_id -> все id разметки с тем же содержимым (по KNOWN_VIDEOS из scripts/evaluate.py)."""
    import ast

    src = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    start = src.index("KNOWN_VIDEOS: dict")
    body = src[src.index("{", start): src.index("\n}\n", start) + 2]
    groups: dict[str, tuple[str, ...]] = {}
    for ids in ast.literal_eval(body).values():
        for vid in ids:
            groups[vid] = tuple(ids)
    return groups


@dataclass
class Truth:
    timecodes: list[float] = field(default_factory=list)
    typed: list[tuple[float, str]] = field(default_factory=list)     # (t, rule_id) — строки с rule_id


def load_truth(path: Path) -> dict[str, Truth]:
    truth: dict[str, Truth] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            vid = normalize_video_id(row.get("video_id") or "")
            if not vid:
                continue
            item = truth.setdefault(vid, Truth())
            raw = (row.get("t_label_sec") or "").strip()
            if not raw:
                continue
            t = float(raw.replace(",", "."))
            item.timecodes.append(t)
            rule_id = (row.get("rule_id") or "").strip()
            if rule_id:
                item.typed.append((t, rule_id))
    return truth


@dataclass
class VideoResult:
    video_id: str
    aliases: list[str]
    truth_timecodes: list[float]
    truth_typed: list[tuple[float, str]]
    events: list[dict]                      # то, что попало бы в ответ
    raw_events: int                         # событий до порога и лимита
    error: Optional[str] = None

    @property
    def truth_violation(self) -> bool:
        return bool(self.truth_timecodes)

    @property
    def pred_violation(self) -> bool:
        return bool(self.events)

    @property
    def outcome(self) -> str:
        if self.error:
            return "ERR"
        if self.truth_violation:
            return "TP" if self.pred_violation else "FN"
        return "FP" if self.pred_violation else "TN"


def evaluate(cache: Path, truth: dict[str, Truth], rules: list, tolerance: float) -> list[VideoResult]:
    groups = known_aliases()
    results: list[VideoResult] = []
    seen: set[tuple[str, ...]] = set()
    for path in sorted(cache.glob("*.json"), key=lambda p: normalize_video_id(p.stem)):
        if path.name.startswith("_"):
            continue
        vid = normalize_video_id(path.stem)
        aliases = groups.get(vid, (vid,))
        if aliases in seen:
            continue                                    # дубликат по содержимому — уже посчитан
        seen.add(aliases)
        tcs = sorted({t for a in aliases for t in (truth[a].timecodes if a in truth else [])})
        typed = sorted({x for a in aliases for x in (truth[a].typed if a in truth else [])})
        try:
            scene = load_scene(path)
            raw = detect_events(rules, scene)
            kept = top_events(raw, rules)
            events = [{"rule_id": ev.rule_id, "t": ev.timecode_sec, "confidence": ev.confidence,
                       "what_visible": ev.facts.get("what_visible")} for ev, _ in kept]
            results.append(VideoResult("+".join(aliases), list(aliases), tcs, typed, events, len(raw)))
        except Exception as exc:  # noqa: BLE001 — одно видео не останавливает замер
            results.append(VideoResult("+".join(aliases), list(aliases), tcs, typed, [], 0,
                                       error=f"{type(exc).__name__}: {exc}"))
    return results


def _near(events: list[dict], t: float, tolerance: float, rule_id: Optional[str] = None) -> bool:
    return any(abs(e["t"] - t) <= tolerance and (rule_id is None or e["rule_id"] == rule_id) for e in events)


def summarize(results: list[VideoResult], rules: list, tolerance: float) -> dict:
    ok = [r for r in results if not r.error]
    counts = {k: sum(1 for r in results if r.outcome == k) for k in ("TP", "FN", "FP", "TN", "ERR")}
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    truth_events = [(r, t) for r in ok for t in r.truth_timecodes]
    typed_rows = [(r, t, rid) for r in ok for t, rid in r.truth_typed]
    per_rule: dict[str, dict[str, int]] = {rule.id: defaultdict(int) for rule in rules}
    for r in ok:
        fired = {e["rule_id"] for e in r.events}
        for rid in fired:
            per_rule.setdefault(rid, defaultdict(int))
            per_rule[rid]["on_violation" if r.truth_violation else "false_alarm_clean"] += 1
        for e in r.events:
            per_rule.setdefault(e["rule_id"], defaultdict(int))
            if e["t"] <= EARLY_SEC:
                per_rule[e["rule_id"]]["early_events"] += 1
        for t in r.truth_timecodes:
            for rid in {e["rule_id"] for e in r.events if abs(e["t"] - t) <= tolerance}:
                per_rule[rid]["timecode_hits"] += 1
    return {
        "videos": len(results),
        "violation_videos": sum(1 for r in ok if r.truth_violation),
        "clean_videos": sum(1 for r in ok if not r.truth_violation),
        **counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else None,
        "tolerance_sec": tolerance,
        "truth_events": len(truth_events),
        "timecode_hits": sum(1 for r, t in truth_events if _near(r.events, t, tolerance)),
        "typed_rows": len(typed_rows),
        "type_hits": sum(1 for r, t, rid in typed_rows if _near(r.events, t, tolerance, rid)),
        "per_rule": {rid: dict(v) for rid, v in sorted(per_rule.items())},
        "missed": [r.video_id for r in ok if r.outcome == "FN"],
        "false_alarms": [r.video_id for r in ok if r.outcome == "FP"],
    }


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def print_report(results: list[VideoResult], s: dict) -> None:
    print(f"{'видео':<14} {'итог':<4} {'эталон, с':<12} события (правило@с conf)")
    for r in results:
        tt = ",".join(f"{t:g}" for t in r.truth_timecodes) or "—"
        ev = "; ".join(f"{e['rule_id'].split('_')[0]}@{e['t']:g} {e['confidence']:.2f}" for e in r.events) or "—"
        print(f"{r.video_id[:14]:<14} {r.outcome:<4} {tt[:12]:<12} {ev}")
        if r.error:
            print(f"    ошибка: {r.error}")
    print()
    print(f"Видео: {s['videos']} (с нарушением {s['violation_videos']}, чистых {s['clean_videos']}, ошибок {s['ERR']})")
    print(f"Вердикт: TP={s['TP']} FN={s['FN']} FP={s['FP']} TN={s['TN']}  верных {_pct(s['accuracy'])}, "
          f"precision {_pct(s['precision'])}, recall {_pct(s['recall'])}")
    print(f"Таймкод ±{s['tolerance_sec']:g} с: {s['timecode_hits']} из {s['truth_events']}; "
          f"тип совпал: {s['type_hits']} из {s['typed_rows']} строк с rule_id")
    print(f"Пропущены: {', '.join(s['missed']) or '—'}")
    print(f"Ложные тревоги: {', '.join(s['false_alarms']) or '—'}")
    print()
    print(f"{'правило':<34} {'ложн.':>5} {'на нар.':>7} {'в ±tol':>6} {'≤1,5 с':>6}")
    for rid, v in s["per_rule"].items():
        print(f"{rid:<34} {v.get('false_alarm_clean', 0):>5} {v.get('on_violation', 0):>7} "
              f"{v.get('timecode_hits', 0):>6} {v.get('early_events', 0):>6}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Замер правил по кэшу сцен без GPU")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "scene_cache")
    ap.add_argument("--annotations", type=Path, default=ROOT / "dataset" / "annotations.csv")
    ap.add_argument("--rules", type=Path, default=ROOT / "rules.yaml")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--json", type=Path, default=None, help="сохранить сводку и результаты по видео")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not args.cache.is_dir():
        print(f"нет каталога кэша {args.cache}", file=sys.stderr)
        return 2
    rules = load_rules(args.rules)
    truth = load_truth(args.annotations)
    results = evaluate(args.cache, truth, rules, args.tolerance)
    if not results:
        print(f"в {args.cache} нет сцен *.json", file=sys.stderr)
        return 2
    summary = summarize(results, rules, args.tolerance)
    print_report(results, summary)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "videos": [{**asdict(r), "outcome": r.outcome} for r in results]}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
