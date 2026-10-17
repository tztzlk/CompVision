#!/usr/bin/env python3
"""Проба ViTPose на трёх кадрах (машина с GPU): модель грузится с диска, точки ставятся, время и VRAM.

    python scripts/download_models.py --only vitpose
    PYTHONPATH=. python scripts/smoke_pose.py --video dataset/videos/Х-20.mov --scene data/scene_cache/Х-20.json
    PYTHONPATH=. python scripts/smoke_pose.py --video dataset/videos/Х-44.mov --times 15 16 17

Рамки людей — из сцены (cashier_bbox и others ближайшего состояния); без --scene — весь кадр
как одна рамка. Печатает версию transformers, время загрузки и инференса, пик видеопамяти,
число видимых точек и координаты запястий; пишет кадры со скелетами в data/viz_pose/smoke_<id>.jpg.
Код возврата: 0 — точки получены хотя бы для одной рамки; 1 — нет; 2 — ошибка окружения.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

MIN_TRANSFORMERS = (4, 48)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for p in text.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Проба ViTPose на трёх кадрах.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--scene", type=Path, help="data/scene_cache/<id>.json — рамки людей")
    p.add_argument("--times", type=float, nargs="+", help="моменты, с (по умолчанию 3 состояния с кассиром)")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--out", type=Path)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"нет torch/transformers: {exc}", file=sys.stderr)
        return 2
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, cuda={torch.cuda.is_available()}")
    if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS:
        print(f"нужен transformers >= {'.'.join(map(str, MIN_TRANSFORMERS))} (VitPoseForPoseEstimation)", file=sys.stderr)
        return 2

    import cv2
    import numpy as np

    from viz_pose import CASHIER_COLOR, OTHER_COLOR, draw_skeleton
    from worker.perception.pose import WRISTS, load_pose_model, visible
    from worker.perception.video import imwrite_unicode, read_frames_at
    from worker.scene.contract import load_scene

    scene = load_scene(args.scene) if args.scene else None
    if args.times:
        times = list(args.times)
    elif scene is not None:
        with_cashier = [s.t for s in scene.states if s.cashier_bbox is not None]
        times = [with_cashier[int(k * (len(with_cashier) - 1) / 2)] for k in range(3)] if with_cashier else [1.0, 2.0, 3.0]
    else:
        times = [1.0, 2.0, 3.0]
    boxes = []
    for t in times:
        st = scene.state_at(t) if scene is not None else None
        boxes.append([st.cashier_bbox, *(p.bbox for p in st.others)] if st is not None else [(0.0, 0.0, 1.0, 1.0)])

    frames = read_frames_at(args.video, times)
    if any(f is None for f in frames):
        print(f"не все кадры прочитаны: {[t for t, f in zip(times, frames) if f is None]}", file=sys.stderr)
    started = time.monotonic()
    estimator = load_pose_model(args.models_dir, args.device)
    print(f"модель загружена за {time.monotonic() - started:.1f} с, device={estimator.device}, dtype={estimator.dtype}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    estimator([f for f in frames if f is not None][:1], [boxes[0]])          # прогрев
    started = time.monotonic()
    result = estimator(frames, boxes)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    n_boxes = sum(1 for bs in boxes for b in bs if b is not None)
    print(f"инференс: {n_boxes} рамок на {len(frames)} кадрах за {elapsed * 1000:.0f} мс")
    if torch.cuda.is_available():
        print(f"пик видеопамяти позы: {torch.cuda.max_memory_allocated() / 2**20:.0f} МБ")

    ok = 0
    tiles = []
    for t, frame, bs, kps in zip(times, frames, boxes, result):
        for j, (box, kp) in enumerate(zip(bs, kps)):
            who = "кассир" if j == 0 and scene is not None else f"человек {j}"
            if kp is None:
                print(f"  t={t:.2f} {who}: точек нет (рамка {box})")
                continue
            ok += 1
            n_vis = sum(1 for i in range(17) if visible(kp, i))
            wr = [(round(kp.points[i][0], 3), round(kp.points[i][1], 3), round(kp.points[i][2], 2)) for i in WRISTS]
            print(f"  t={t:.2f} {who}: видимых точек {n_vis}/17, запястья (x, y, score) {wr}")
        if frame is not None:
            tile = frame.copy()
            for j, kp in enumerate(kps):
                draw_skeleton(tile, kp, CASHIER_COLOR if j == 0 else OTHER_COLOR)
            tiles.append(tile)
    if tiles:
        h = min(t.shape[0] for t in tiles)
        sheet = np.hstack([cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles])
        out = args.out or ROOT / "data" / "viz_pose" / f"smoke_{args.video.stem}.jpg"
        imwrite_unicode(out, sheet, quality=85)
        print(f"кадры со скелетами: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
