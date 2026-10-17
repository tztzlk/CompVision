#!/usr/bin/env python3
"""Где срабатывает одно правило с подменёнными параметрами — по кэшу сцен, без GPU.

    PYTHONPATH=. python scripts/sweep_rule.py --cache data/scene_cache_r3 R10 baseline_delta=0.25 min_duration_sec=7

Значения параметров — JSON (числа, true/false, "строки" в кавычках). Печатает видео, где правило дало
событие выше min_confidence, с эталоном, и итог: на скольких видео с нарушением и на скольких чистых.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_cache import known_aliases, load_truth, normalize_video_id  # noqa: E402
from worker.events import detect_events, load_rules  # noqa: E402
from worker.scene.contract import load_scene  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rules", type=Path, default=ROOT / "rules.yaml")
    ap.add_argument("--annotations", type=Path, default=ROOT / "dataset" / "annotations.csv")
    ap.add_argument("rule", help="начало id правила, например R10")
    ap.add_argument("params", nargs="*", help="key=value (value — JSON)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    over = {}
    for kv in args.params:
        key, value = kv.split("=", 1)
        over[key] = json.loads(value)
    rules = [dataclasses.replace(r, params={**r.params, **over})
             for r in load_rules(args.rules) if r.id.startswith(args.rule)]
    if not rules:
        print(f"нет правила {args.rule}", file=sys.stderr)
        return 2
    min_conf = float(rules[0].params.get("min_confidence", 0.5))
    truth, groups = load_truth(args.annotations), known_aliases()
    seen: set[tuple[str, ...]] = set()
    hit = false = 0
    for path in sorted(args.cache.glob("*.json"), key=lambda p: normalize_video_id(p.stem)):
        if path.name.startswith("_"):
            continue
        vid = normalize_video_id(path.stem)
        aliases = groups.get(vid, (vid,))
        if aliases in seen:
            continue
        seen.add(aliases)
        label = sorted({t for a in aliases for t in (truth[a].timecodes if a in truth else [])})
        events = [e for e in detect_events(rules, load_scene(path)) if e.confidence >= min_conf]
        if not events:
            continue
        hit += bool(label)
        false += not label
        shown = ", ".join(f"{e.timecode_sec:g} ({e.confidence:.2f})" for e in events[:5])
        print(f"{'нарушение' if label else 'ЧИСТОЕ':9s} {'+'.join(aliases):10s} эталон {label or '—'}: {shown}")
    print(f"срабатывает на видео с нарушением: {hit}, на чистых: {false}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
