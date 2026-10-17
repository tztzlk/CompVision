#!/usr/bin/env python3
"""Листы для разметки жестов кассира глазами: вырезка вокруг кассира, стола и сейфа, скелеты, время.

    PYTHONPATH=. python scripts/pose_label_sheet.py --scenes data/scene_cache_r3_pose
    PYTHONPATH=. python scripts/pose_label_sheet.py --scenes ... --only Х-20 --every 1 --cols 4 --rows 3

Плитка — кадр состояния, ближайшего к целой секунде (шаг --every), обрезанный до области работы
кассира: объединение зон стола и сейфа, рамок кассира и людей в комнате (одна область на всё видео,
чтобы плитки было легко сравнивать). Скелет кассира — зелёный, остальных людей — красный, запястья —
кружки, зона сейфа — синяя рамка; вверху плитки — время «t=12». Кадры — из data/frames_r2/<video_id>/
(или --video, если кадров нет). Подписи латиницей: шрифты OpenCV не поддерживают кириллицу.

Метки по этим листам — training/labels/pose_gestures.csv (scripts/train_pose_heads.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from worker.perception.pose import SKELETON, WRISTS, Keypoints  # noqa: E402
from worker.perception.video import imread_unicode, imwrite_unicode, read_frames_at  # noqa: E402
from worker.scene.contract import SceneState, SceneTimeline, load_scene  # noqa: E402

CASHIER_COLOR = (0, 255, 0)          # BGR
OTHER_COLOR = (0, 0, 255)
SAFE_COLOR = (255, 128, 0)
MIN_SCORE = 0.3
MARGIN = 0.04                        # запас области по краям, доля кадра


def _kp(raw: Any) -> Optional[Keypoints]:
    try:
        return Keypoints.from_list(raw)
    except (ValueError, TypeError, IndexError):
        return None


def latin(text: str) -> str:
    """Кириллическая «Х» в номере видео — латиницей (OpenCV не рисует кириллицу)."""
    return text.replace("Х", "X").replace("х", "x")


def work_region(scene: SceneTimeline) -> tuple[float, float, float, float]:
    """Область работы кассира на всё видео (доли кадра): зоны стола и сейфа + рамки людей в комнате."""
    boxes = [z for name, z in scene.zones.items() if name in ("desk", "safe", "workplace") and z is not None]
    for s in scene.states:
        if s.cashier_bbox is not None:
            boxes.append(s.cashier_bbox)
        boxes.extend(p.bbox for p in s.others)
    if not boxes:
        return (0.0, 0.0, 1.0, 1.0)
    x0 = max(0.0, min(b[0] for b in boxes) - MARGIN)
    y0 = max(0.0, min(b[1] for b in boxes) - MARGIN)
    x1 = min(1.0, max(b[2] for b in boxes) + MARGIN)
    y1 = min(1.0, max(b[3] for b in boxes) + MARGIN)
    return (x0, y0, x1, y1)


def pick_states(scene: SceneTimeline, every: float) -> list[SceneState]:
    """Состояние, ближайшее к каждой отметке 0, every, 2·every … (без повторов)."""
    out: list[SceneState] = []
    if not scene.states:
        return out
    t, last = 0.0, scene.states[-1].t
    while t <= last + 1e-6:
        best = min(scene.states, key=lambda s: abs(s.t - t))
        if abs(best.t - t) <= every / 2 + 1e-6 and (not out or best is not out[-1]):
            out.append(best)
        t += every
    return out


def _draw_skeleton(img: np.ndarray, kp: Optional[Keypoints], color: tuple[int, int, int],
                   region: tuple[float, float, float, float], frame_size: tuple[int, int], k: float) -> None:
    if kp is None:
        return
    w, h = frame_size
    pts = [(int(((x - region[0]) * w) * k), int(((y - region[1]) * h) * k), sc) for x, y, sc in kp.points]
    for a, b in SKELETON:
        if a < 5 or b < 5:
            continue                       # точки головы не рисуем: лицо не размечаем (FAQ 29)
        if pts[a][2] >= MIN_SCORE and pts[b][2] >= MIN_SCORE:
            cv2.line(img, pts[a][:2], pts[b][:2], color, 1, cv2.LINE_AA)
    for i in WRISTS:
        x, y, sc = pts[i]
        if sc >= MIN_SCORE:
            cv2.circle(img, (x, y), 7, color, 1, cv2.LINE_AA)


def render_tile(image: Optional[np.ndarray], scene: SceneTimeline, state: SceneState,
                region: tuple[float, float, float, float], tile_w: int, tile_h: int) -> np.ndarray:
    w, h = scene.frame_size
    rx0, ry0, rx1, ry1 = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)
    if image is None:
        crop = np.full((max(1, ry1 - ry0), max(1, rx1 - rx0), 3), 64, dtype=np.uint8)
    else:
        if image.shape[1] != w or image.shape[0] != h:
            image = cv2.resize(image, (w, h))
        crop = image[ry0:ry1, rx0:rx1].copy()
    k = min(tile_w / max(1, crop.shape[1]), tile_h / max(1, crop.shape[0]))
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))),
                      interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    safe = scene.zones.get("safe")
    if safe is not None:
        p0 = (int((safe[0] - region[0]) * w * k), int((safe[1] - region[1]) * h * k))
        p1 = (int((safe[2] - region[0]) * w * k), int((safe[3] - region[1]) * h * k))
        cv2.rectangle(crop, p0, p1, SAFE_COLOR, 1, cv2.LINE_AA)
    _draw_skeleton(crop, _kp(state.value("cashier_kp")), CASHIER_COLOR, region, scene.frame_size, k)
    for item in state.value("others_kp") or []:
        if isinstance(item, dict):
            _draw_skeleton(crop, _kp(item.get("kp")), OTHER_COLOR, region, scene.frame_size, k)
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile[:crop.shape[0], :crop.shape[1]] = crop
    label = f"t={state.t:.0f}"
    cv2.rectangle(tile, (0, 0), (8 + 13 * len(label), 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if state.cashier_bbox is None:
        cv2.putText(tile, "no cashier box", (4, tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return tile


def render_sheets(scene: SceneTimeline, frames: Sequence[Optional[np.ndarray]], states: Sequence[SceneState],
                  cols: int, rows: int, tile_w: int, tile_h: int) -> list[np.ndarray]:
    region = work_region(scene)
    tiles = [render_tile(img, scene, st, region, tile_w, tile_h) for img, st in zip(frames, states)]
    per = cols * rows
    sheets: list[np.ndarray] = []
    for start in range(0, len(tiles), per):
        chunk = tiles[start:start + per]
        blank = np.zeros_like(chunk[0])
        while len(chunk) % cols:
            chunk.append(blank)
        grid = [np.hstack([cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                           for t in chunk[i:i + cols]]) for i in range(0, len(chunk), cols)]
        sheets.append(np.vstack(grid))
    return sheets


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Листы для разметки жестов кассира (скелеты, время).")
    p.add_argument("--scenes", type=Path, default=ROOT / "data" / "scene_cache_r3_pose", help="сцены с позой")
    p.add_argument("--frames", type=Path, default=ROOT / "data" / "frames_r2", help="<video_id>/<кадр>.jpg")
    p.add_argument("--videos", type=Path, default=ROOT / "dataset" / "videos", help="если кадров нет")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "ml_results" / "pose" / "label_sheets")
    p.add_argument("--only", action="append", metavar="VIDEO_ID")
    p.add_argument("--every", type=float, default=1.0, help="шаг, с")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--tile-w", type=int, default=380)
    p.add_argument("--tile-h", type=int, default=300)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    paths = sorted(args.scenes.glob("*.json"))
    if args.only:
        paths = [p for p in paths if p.stem in set(args.only)]
    if not paths:
        print(f"нет сцен в {args.scenes}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        scene = load_scene(path)
        states = pick_states(scene, args.every)
        frames_dir = args.frames / scene.video_id
        if frames_dir.is_dir():
            frames = [imread_unicode(frames_dir / Path(s.frame).name) for s in states]
        else:
            frames = read_frames_at(args.videos / f"{scene.video_id}.mov", [s.t for s in states])
        sheets = render_sheets(scene, frames, states, args.cols, args.rows, args.tile_w, args.tile_h)
        for i, sheet in enumerate(sheets, 1):
            imwrite_unicode(args.out / f"{latin(scene.video_id)}_{i:02d}.jpg", sheet, quality=90)
        print(f"{scene.video_id}: {len(states)} плиток, {len(sheets)} листов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
