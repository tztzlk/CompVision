#!/usr/bin/env python3
"""Таблица устойчивости: одни и те же правила на кэшах оригинала и аугментаций (scripts/augment_videos.py).

    PYTHONPATH=. .venv/bin/python scripts/robustness_table.py \
        оригинал=data/scene_cache_r3 flip=data/scene_cache_aug_flip persp=data/scene_cache_aug_persp ...

Первый кэш — база. Печатает markdown: найдено / ложных / верный вердикт / таймкод ±3 с по каждому кэшу;
по правилам — «видео с нарушением, где правило сработало в ±3 с от эталона» и «ложных на чистых»;
списки видео, сменивших вердикт против базы. --json — то же в файл.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cache import evaluate, load_truth, summarize  # noqa: E402
from worker.events import load_rules  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Таблица устойчивости правил к аугментациям.")
    ap.add_argument("caches", nargs="+", help="имя=каталог кэша сцен; первый — база")
    ap.add_argument("--annotations", type=Path, default=ROOT / "dataset" / "annotations.csv")
    ap.add_argument("--rules", type=Path, default=ROOT / "rules.yaml")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rules = load_rules(args.rules)
    truth = load_truth(args.annotations)
    names, summaries, outcomes = [], [], []
    for spec in args.caches:
        name, _, path = spec.partition("=")
        results = evaluate(Path(path), truth, rules, args.tolerance)
        names.append(name)
        summaries.append(summarize(results, rules, args.tolerance))
        outcomes.append({r.video_id: r.outcome for r in results})

    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    rows = [
        ("найдено", lambda s: f"{s['TP']} из {s['TP'] + s['FN']}"),
        ("ложных", lambda s: f"{s['FP']} из {s['FP'] + s['TN']}"),
        ("верный вердикт", lambda s: f"{(s['accuracy'] or 0) * 100:.0f}%"),
        ("таймкод ±3 с", lambda s: f"{s['timecode_hits']} из {s['truth_events']}"),
        ("ошибок анализа", lambda s: str(s["ERR"])),
    ]
    for label, fn in rows:
        print(f"| {label} | " + " | ".join(fn(s) for s in summaries) + " |")
    print("\nпо правилам: попаданий ±3 с / ложных на чистых")
    print("| правило | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    rule_ids = sorted({rid for s in summaries for rid in s["per_rule"]})
    for rid in rule_ids:
        cells = []
        for s in summaries:
            r = s["per_rule"].get(rid, {})
            cells.append(f"{r.get('timecode_hits', 0)} / {r.get('false_alarm_clean', 0)}")
        print(f"| {rid} | " + " | ".join(cells) + " |")
    base = outcomes[0]
    print()
    for name, out in zip(names[1:], outcomes[1:]):
        lost = sorted(v for v, o in out.items() if base.get(v) == "TP" and o != "TP")
        gained = sorted(v for v, o in out.items() if base.get(v) == "FN" and o == "TP")
        new_fp = sorted(v for v, o in out.items() if o == "FP" and base.get(v) != "FP")
        print(f"{name}: потеряно {', '.join(lost) or '—'}; найдено новых {', '.join(gained) or '—'}; "
              f"новые ложные {', '.join(new_fp) or '—'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"names": names, "summaries": summaries, "outcomes": outcomes},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
