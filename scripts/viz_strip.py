#!/usr/bin/env python3
"""Раскадровка одного видео по кэшу сцен: зоны, кассир (зелёный), другие внутри (жёлтые), все сырые треки
(серые с id) — для разбора отдельных моментов.

    PYTHONPATH=. python scripts/viz_strip.py --scene data/scene_cache_r3/Х-26.json --raw data/raw_cache_r3/Х-26.json \\
        --frames data/frames_r2/Х-26 --out strip.jpg 8 9.3 12 14
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

from worker.perception.video import imread_unicode, imwrite_unicode  # noqa: E402
from worker.scene.contract import load_scene  # noqa: E402

COLORS = {"desk": (0, 200, 0), "safe": (0, 0, 255), "door": (255, 0, 0), "laptop": (255, 0, 255),
          "workplace": (0, 255, 255)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("times", type=float, nargs="+")
    args = ap.parse_args(argv)
    sc = load_scene(args.scene)
    raw = json.loads(args.raw.read_text(encoding="utf-8")) if args.raw else {"tracks": []}
    tiles = []
    for t in args.times:
        i = min(range(len(sc.states)), key=lambda k: abs(sc.states[k].t - t))
        s = sc.states[i]
        img = imread_unicode(args.frames / Path(s.frame).name)
        h, w = img.shape[:2]

        def rect(b, c, th, lab=None):
            x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, th)
            if lab:
                cv2.putText(img, lab, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        for name, b in sc.zones.items():
            if b:
                rect(b, COLORS.get(name, (128, 128, 128)), 1, name)
        for tr in raw["tracks"]:
            for p in tr["points"]:
                if p[0] == i:
                    rect(p[2], (160, 160, 160), 1, f"{tr['id']}:{p[3]:.2f}")
        if s.cashier_bbox:
            rect(s.cashier_bbox, (0, 255, 0), 3, "C")
        for p in s.others:
            rect(p.bbox, (0, 220, 255), 2, f"o{p.track_id}")
        cv2.putText(img, f"t={s.t:.1f} n={s.persons_in_room} pr={s.cashier_present}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(img)
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[k:k + args.cols] + [blank] * (args.cols - len(tiles[k:k + args.cols])))
            for k in range(0, len(tiles), args.cols)]
    out = np.vstack(rows)
    scale = min(1.0, 1800 / out.shape[1])
    imwrite_unicode(args.out, cv2.resize(out, None, fx=scale, fy=scale), 85)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
