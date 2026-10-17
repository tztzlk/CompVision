#!/usr/bin/env python3
"""Где срабатывают детекторы позы: события на чистых видео (ложные тревоги) и у меток нарушений.

    PYTHONPATH=. python scripts/eval_pose_detectors.py --cache data/scene_cache_r3_pose
    PYTHONPATH=. python scripts/eval_pose_detectors.py --cache data/scene_cache_r3_pose data/scene_cache_r3_pose_heads

Детекторы — worker.events.pose_detectors.POSE_DETECTORS с параметрами по умолчанию (или --params,
JSON {детектор: {параметр: значение}}). Разметка — training/labels/violation_types.csv (формат
dataset/annotations.csv, типы — из проверенной таблицы команды): для каждой метки нарушения — есть ли
событие детектора в ±--tolerance с и связан ли тип метки с детектором (RELATED).
Правила с детекторами позы в rules.yaml выключены: таблица показывает, что было бы при включении.
Копия Х-48 (≡ Х-5) не считается; Х-47 (≡ Х-4) остаётся — у него своя метка (68 с).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.events.pose_detectors import POSE_DETECTORS  # noqa: E402
from worker.scene.contract import load_scene  # noqa: E402

SKIP = {"Х-48"}
# Типы нарушений (rule_id разметки), которые детектор позы мог бы подтвердить.
RELATED: dict[str, set[str]] = {
    "hand_to_hand_transfer": {"R11_outsider_in_cash_room", "R12_collection_inside", "R17_accept_without_id"},
    "counting_without_controller": {"R03_accept_without_count", "R08_count_without_controller"},
    "drink_or_phone_at_face": {"R13_drink_at_workplace"},
    "storage_left_open_pose": {"R10_storage_left_open"},
}


def load_annotations(path: Path) -> tuple[dict[str, list[tuple[float, str]]], set[str]]:
    """(метки нарушений по видео [(t, rule_id)], чистые видео)."""
    labels: dict[str, list[tuple[float, str]]] = {}
    clean: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_id"].strip()
            t = row["t_label_sec"].strip()
            if t:
                labels.setdefault(vid, []).append((float(t), row.get("rule_id", "").strip()))
            else:
                clean.add(vid)
    return labels, clean


def run(cache: Path, annotations: Path, params: dict[str, Any], tolerance: float) -> dict[str, Any]:
    labels, clean = load_annotations(annotations)
    out: dict[str, Any] = {name: {"clean_fired": [], "label_hits": [], "type_hits": [], "events": 0}
                           for name in POSE_DETECTORS}
    for path in sorted(cache.glob("*.json")):
        vid = path.stem
        if vid in SKIP or path.name.startswith("_"):
            continue
        scene = load_scene(path)
        for name, fn in POSE_DETECTORS.items():
            events = fn(scene, params.get(name, {}))
            out[name]["events"] += len(events)
            if vid in clean and events:
                out[name]["clean_fired"].append(f"{vid}@{events[0].timecode_sec:.1f}")
            for t, rule_id in labels.get(vid, []):
                near = [e for e in events if abs(e.timecode_sec - t) <= tolerance]
                if near:
                    out[name]["label_hits"].append(f"{vid}@{t:g}")
                    if rule_id in RELATED[name]:
                        out[name]["type_hits"].append(f"{vid}@{t:g}")
    out["_n_clean"] = len(clean - SKIP)
    out["_n_labels"] = sum(len(v) for v in labels.values())
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Детекторы позы на кэше сцен: ложные тревоги и попадания.")
    p.add_argument("--cache", type=Path, nargs="+", required=True, help="кэши сцен с позой")
    p.add_argument("--annotations", type=Path, default=ROOT / "training" / "labels" / "violation_types.csv")
    p.add_argument("--params", default="{}", help="JSON {детектор: {параметр: значение}}")
    p.add_argument("--tolerance", type=float, default=3.0)
    p.add_argument("--json", type=Path, default=None, help="сохранить результат")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    params = json.loads(args.params)
    result: dict[str, Any] = {}
    print("| кэш | детектор | событий | чистые с событием | у метки ±3 с | тип совпал |")
    print("|---|---|---|---|---|---|")
    for cache in args.cache:
        res = run(cache, args.annotations, params, args.tolerance)
        result[str(cache)] = res
        for name in POSE_DETECTORS:
            r = res[name]
            print(f"| {cache.name} | {name} | {r['events']} | {len(r['clean_fired'])}/{res['_n_clean']} "
                  f"{' '.join(r['clean_fired'])} | {len(r['label_hits'])}/{res['_n_labels']} {' '.join(r['label_hits'])} | "
                  f"{' '.join(r['type_hits']) or '—'} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
