#!/usr/bin/env python3
"""Перенести точки позы из кэша сцен с позой в сырой кэш: поле person_pose для builder (без моделей).

    PYTHONPATH=. python scripts/add_pose_to_raw.py --raw data/ml_results/dep/raw_base \
        --pose-cache data/scene_cache_r3_pose --out data/ml_results/inside_ft/pose/raw_pose

Точки (ViTPose, scripts/add_pose_to_cache.py) лежат в полях состояний сцены: cashier_kp — по рамке
кассира, others_kp — по людям из others. Состояние сопоставляется с кадром сырого кэша по времени
в мс (int(round(t * 1000))), рамка — та, что в сцене (builder ищет её по IoU, worker.scene.pose_departure).
В сырой кэш пишется person_pose = [[позиция кадра, рамка, [[x, y, score] x 17]], ...]; остальное — как было.
Видео без сцены с позой копируется без поля. Сырые кэши аугментаций — так же (--raw data/raw_cache_aug_flip).

Замечание: точки посчитаны только для рамок, которые попали в сцену (кассир после вычитания ухода и
others внутри кассы) — на кадрах «вдали» (кассир за стеклом) точек нет.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]


def norm_id(video_id: str) -> str:
    """Латинская X- и кириллическая Х- — одно и то же."""
    s = unicodedata.normalize("NFC", video_id).strip()
    return s.replace("Х", "X").replace("х", "x").upper()


def ms(t: float) -> int:
    return int(round(float(t) * 1000))


def pose_entries(raw: dict[str, Any], scene: dict[str, Any]) -> tuple[list[list[Any]], dict[str, int]]:
    """person_pose для сырого кэша по сцене с позой и счётчики покрытия."""
    pos_by_ms = {ms(t): pos for pos, (_i, t, _f) in enumerate(raw["frames"])}
    out: list[list[Any]] = []
    stats = {"states": len(scene["states"]), "matched_t": 0, "cashier_kp": 0, "others_kp": 0}
    for st in scene["states"]:
        pos = pos_by_ms.get(ms(st["t"]))
        if pos is None:
            continue
        stats["matched_t"] += 1
        extra = st.get("extra") or st      # сцена хранит extra-поля на верхнем уровне состояния
        kp = extra.get("cashier_kp")
        if kp and st.get("cashier_bbox"):
            out.append([pos, list(st["cashier_bbox"]), kp])
            stats["cashier_kp"] += 1
        boxes = {o.get("track_id"): o.get("bbox") for o in st.get("others") or []}
        for o in extra.get("others_kp") or []:
            box = boxes.get(o.get("track_id"))
            if o.get("kp") and box:
                out.append([pos, list(box), o["kp"]])
                stats["others_kp"] += 1
    return out, stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Поза из кэша сцен -> поле person_pose сырого кэша")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pose-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    scenes = {norm_id(p.stem): p for p in args.pose_cache.glob("*.json") if not p.name.startswith("_")}
    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    if not files:
        print(f"в {args.raw} нет сырого кэша *.json", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    total = {"videos": 0, "with_pose": 0, "cashier_kp": 0, "others_kp": 0}
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("person_pose", None)
        scene_path = scenes.get(norm_id(path.stem))
        total["videos"] += 1
        if scene_path is not None:
            entries, stats = pose_entries(raw, json.loads(scene_path.read_text(encoding="utf-8")))
            if entries:
                raw["person_pose"] = entries
                total["with_pose"] += 1
            total["cashier_kp"] += stats["cashier_kp"]
            total["others_kp"] += stats["others_kp"]
            print(f"{path.stem}: кадров {len(raw['frames'])}, состояний {stats['states']} "
                  f"(по времени совпало {stats['matched_t']}), точек кассира {stats['cashier_kp']}, "
                  f"других {stats['others_kp']}")
        else:
            print(f"{path.stem}: сцены с позой нет — без person_pose")
        dst = args.out / path.name
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dst)
    print(f"видео {total['videos']}, с позой {total['with_pose']}, точек кассира {total['cashier_kp']}, "
          f"других {total['others_kp']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
