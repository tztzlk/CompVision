#!/usr/bin/env python3
"""Досчитать в готовый сырой кэш сигнал двери для R15 (door_diff) — без моделей и GPU, только кадры.

    PYTHONPATH=. python scripts/add_door_to_raw.py --raw data/raw_cache_r3 --frames data/frames_r3 [--only Х-32]

Кадры (--frames/<video_id>/<номер>.jpg) — те же, что для scripts/rezone_raw.py. Зона двери и люди
(закрашиваются) берутся из сырого кэша. Файл кэша переписывается на месте (поле door_diff); дальше —
scripts/rebuild_from_raw.py и scripts/eval_cache.py. Пороги door_change (SceneConfig.door_diff_floor /
door_diff_span) применяются при пересборке, сюда не входят.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.perception import Detection, Frame  # noqa: E402
from worker.scene.builder import SceneConfig  # noqa: E402
from worker.scene.pipeline import door_signal  # noqa: E402
from worker.scene.raw_cache import load_raw  # noqa: E402


def door_diff_for(raw: dict, frames_dir: Path, cfg: SceneConfig) -> Optional[list[Optional[float]]]:
    """door_diff по сырому кэшу одного видео и его кадрам (None — в кэше нет зоны двери)."""
    door = (raw.get("zones") or {}).get("door")
    frames = [Frame(index=int(i), timestamp_sec=float(t), path=frames_dir / Path(f).name) for i, t, f in raw["frames"]]
    persons: list[list[Detection]] = [[] for _ in frames]
    for tr in raw["tracks"]:
        for fi, _, box, score in tr["points"]:
            if 0 <= int(fi) < len(persons):
                persons[int(fi)].append(Detection("person", float(score), tuple(box)))  # type: ignore[arg-type]
    return door_signal(frames, persons, tuple(door) if door else None, cfg)  # type: ignore[arg-type]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Сигнал двери (R15) в готовый сырой кэш, без моделей")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--only", action="append")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = SceneConfig()
    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    done = 0
    for path in files:
        vid = path.stem
        if args.only and vid not in args.only:
            continue
        raw = load_raw(path)
        diffs = door_diff_for(raw, args.frames / vid, cfg)
        if diffs is None:
            print(f"{vid}: нет зоны двери — пропущено")
            continue
        raw["door_diff"] = [None if d is None else round(float(d), 4) for d in diffs]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        known = [d for d in diffs if d is not None]
        print(f"{vid}: {len(known)} из {len(diffs)} кадров, max {max(known, default=0):.2f}")
        done += 1
    print(f"сигнал двери записан в {done} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
