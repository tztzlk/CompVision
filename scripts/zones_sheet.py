#!/usr/bin/env python3
"""Контактный лист зон: по плитке на видео (кадр + desk, safe, door, laptop) — проверка выбора зон глазами.

    PYTHONPATH=. python scripts/zones_sheet.py --raw data/raw_cache_r2 --frames data/frames_r2 \\
        --out data/ml_results/zones_after.jpg [--owl data/owl_r2] [--rezone]

С --rezone зоны выбираются заново текущим кодом (worker.scene.pipeline.select_zones) по детекциям OWLv2
из --owl (или из сырого кэша); без него — зоны, записанные в сыром кэше.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from worker.perception import Detection  # noqa: E402
from worker.perception.tracking import Track, TrackPoint  # noqa: E402
from worker.perception.video import imread_unicode, imwrite_unicode  # noqa: E402
from worker.scene.builder import SceneConfig  # noqa: E402
from worker.scene.pipeline import select_zones  # noqa: E402

COLORS = {"desk": (0, 200, 0), "safe": (0, 0, 255), "door": (255, 0, 0), "laptop": (255, 0, 255)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--owl", type=Path, default=None)
    ap.add_argument("--rezone", action="store_true")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--only", action="append")
    args = ap.parse_args(argv)
    det = lambda d: Detection(str(d[0]), float(d[1]), tuple(d[2]))  # noqa: E731
    tiles = []
    for path in sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_")):
        vid = path.stem
        if args.only and vid not in args.only:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        zones = raw["zones"]
        why = ""
        if args.rezone:
            owl_file = args.owl / f"{vid}.json" if args.owl else None
            owl_raw = (json.loads(owl_file.read_text(encoding="utf-8"))["owl"] if owl_file and owl_file.is_file()
                       else raw.get("owl_keyframes", []))
            tracks = [Track(int(t["id"]), [TrackPoint(int(a), float(b), tuple(c), float(d)) for a, b, c, d in t["points"]])
                      for t in raw["tracks"]]
            zones, info = select_zones([[det(d) for d in ds] for ds in owl_raw],
                                       [[det(d) for d in ds] for ds in raw["objects"]], tracks, SceneConfig())
            why = str(info.get("safe", {}).get("reason", ""))
        frame = raw["frames"][len(raw["frames"]) // 4][2]
        img = imread_unicode(args.frames / vid / Path(frame).name)
        if img is None:
            continue
        h, w = img.shape[:2]
        for name, box in zones.items():
            if box is None or name not in COLORS:
                continue
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[name], 3 if name in ("desk", "safe") else 1)
            cv2.putText(img, name, (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2)
        cv2.putText(img, vid.replace("Х", "X"), (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if zones.get("safe") is None:
            cv2.putText(img, "no safe " + why[:30], (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(cv2.resize(img[: int(h * 0.75)], (w * 300 // h, int(h * 0.75) * 300 // h)))
    if not tiles:
        return 2
    th, tw = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, tw - t.shape[1] + 3, cv2.BORDER_CONSTANT) for t in tiles]
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[k:k + args.cols] + [blank] * (args.cols - len(tiles[k:k + args.cols])))
            for k in range(0, len(tiles), args.cols)]
    imwrite_unicode(args.out, np.vstack(rows), 85)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
