#!/usr/bin/env python3
"""Контактный лист позы: скелеты людей, признаки позы и сработавшие детекторы позы по кадрам.

    PYTHONPATH=. python scripts/viz_pose.py --scene data/scene_cache_pose/Х-44.json --video dataset/videos/Х-44.mov
    PYTHONPATH=. python scripts/viz_pose.py --scene ... --video ... --times 15 16 17 --cols 3

Сцена — с полями позы (scripts/add_pose_to_cache.py). Поверх листа scripts/viz_scene.py:
скелет кассира — зелёным, остальных людей — красным, запястья — кружками; внизу плитки —
признаки позы, вверху — детекторы позы, событие которых началось за последние --recent с.
Результат — JPEG (по умолчанию data/viz_pose/<video_id>.jpg). Нужен только OpenCV.
Подписи латиницей: шрифты OpenCV не поддерживают кириллицу.
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

from viz_scene import _text, draw_state, select_states  # noqa: E402
from worker.events.model import Event  # noqa: E402
from worker.events.pose_detectors import POSE_DETECTORS  # noqa: E402
from worker.perception.pose import SKELETON, WRISTS, Keypoints  # noqa: E402
from worker.perception.video import imread_unicode, imwrite_unicode, read_frames_at  # noqa: E402
from worker.scene.contract import SceneState, SceneTimeline, load_scene  # noqa: E402

CASHIER_COLOR = (0, 255, 0)          # BGR
OTHER_COLOR = (0, 0, 255)
EVENT_COLOR = (0, 200, 255)
MIN_SCORE = 0.3

FEATURE_LABELS = (
    ("cashier_hand_in_safe", "safe"), ("cashier_hands_on_desk", "desk"), ("hand_to_hand", "h2h"),
    ("cashier_counting", "count"), ("cashier_hand_at_face", "face"), ("cashier_standing", "stand"),
)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _kp(raw: Any) -> Optional[Keypoints]:
    try:
        return Keypoints.from_list(raw)
    except (ValueError, TypeError, IndexError):
        return None


def draw_skeleton(tile: np.ndarray, kp: Optional[Keypoints], color: tuple[int, int, int]) -> None:
    if kp is None:
        return
    h, w = tile.shape[:2]
    pts = [(int(x * w), int(y * h), s) for x, y, s in kp.points]
    for a, b in SKELETON:
        if pts[a][2] >= MIN_SCORE and pts[b][2] >= MIN_SCORE:
            cv2.line(tile, pts[a][:2], pts[b][:2], color, 2, cv2.LINE_AA)
    for i, (x, y, s) in enumerate(pts):
        if s < MIN_SCORE:
            continue
        if i in WRISTS:
            cv2.circle(tile, (x, y), 6, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(tile, (x, y), 2, color, -1, cv2.LINE_AA)


def run_pose_detectors(scene: SceneTimeline) -> list[tuple[str, Event]]:
    """Все детекторы позы с параметрами по умолчанию: [(имя, событие)]."""
    out: list[tuple[str, Event]] = []
    for name, fn in POSE_DETECTORS.items():
        out.extend((name, ev) for ev in fn(scene, {}))
    return sorted(out, key=lambda item: item[1].timecode_sec)


def draw_pose_state(
    image: np.ndarray, scene: SceneTimeline, state: SceneState, tile_width: int,
    events: Sequence[tuple[str, Event]], recent_sec: float,
) -> np.ndarray:
    tile = draw_state(image, scene, state, tile_width)
    draw_skeleton(tile, _kp(state.value("cashier_kp")), CASHIER_COLOR)
    for item in state.value("others_kp") or []:
        if isinstance(item, dict):
            draw_skeleton(tile, _kp(item.get("kp")), OTHER_COLOR)
    values = [f"{label} {_fmt(state.value(name))}" for name, label in FEATURE_LABELS]
    track = state.value("hand_to_hand_track")
    if track is not None:
        values[2] += f"#{track}"
    lines = [" ".join(values[:3]), " ".join(values[3:])]
    for i, line in enumerate(lines):
        _text(tile, line, (4, tile.shape[0] - 40 - 14 * (len(lines) - 1 - i)), CASHIER_COLOR)
    fired = [f"{name}@{ev.timecode_sec:.1f}" for name, ev in events
             if ev.timecode_sec - 1e-6 <= state.t <= ev.timecode_sec + recent_sec]
    for i, label in enumerate(fired[:3]):
        _text(tile, label, (4, 32 + 14 * i), EVENT_COLOR)
    return tile


def render_contact_sheet(
    scene: SceneTimeline,
    frames: Sequence[Optional[np.ndarray]],
    states: Sequence[SceneState],
    *,
    cols: int = 6,
    tile_width: int = 300,
    recent_sec: float = 3.0,
) -> np.ndarray:
    events = run_pose_detectors(scene)
    tiles = []
    for image, state in zip(frames, states):
        if image is None:
            height = int(tile_width * scene.frame_size[1] / max(1, scene.frame_size[0]))
            image = np.full((max(1, height), tile_width, 3), 64, dtype=np.uint8)
        tiles.append(draw_pose_state(image, scene, state, tile_width, events, recent_sec))
    if not tiles:
        raise ValueError("нет кадров для контактного листа")
    tile_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, tile_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Контактный лист позы: скелеты, признаки, детекторы позы.")
    p.add_argument("--scene", type=Path, required=True, help="data/scene_cache_pose/<video_id>.json")
    p.add_argument("--video", type=Path, help="исходное видео (если кадры не сохранены)")
    p.add_argument("--frames-dir", type=Path, help="рабочий каталог с frames/")
    p.add_argument("--times", type=float, nargs="+", help="конкретные моменты, с")
    p.add_argument("--every", type=float, default=1.0, help="шаг по времени, с (если --times не задан)")
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--tile-width", type=int, default=300)
    p.add_argument("--recent", type=float, default=3.0, help="показывать события, начавшиеся за N с")
    p.add_argument("--out", type=Path, help="JPEG (по умолчанию data/viz_pose/<video_id>.jpg)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.video is None and args.frames_dir is None:
        print("нужен --video или --frames-dir", file=sys.stderr)
        return 2
    scene = load_scene(args.scene)
    if not any(s.value("cashier_kp") is not None for s in scene.states):
        print(f"в {args.scene} нет точек позы — сначала scripts/add_pose_to_cache.py", file=sys.stderr)
    states = select_states(scene, args.times, args.every)
    if args.frames_dir is not None:
        frames = [imread_unicode(args.frames_dir / s.frame) for s in states]
    else:
        frames = read_frames_at(args.video, [s.t for s in states])
    sheet = render_contact_sheet(scene, frames, states, cols=args.cols, tile_width=args.tile_width,
                                 recent_sec=args.recent)
    out = args.out or ROOT / "data" / "viz_pose" / f"{scene.video_id}.jpg"
    imwrite_unicode(out, sheet, quality=85)
    print(f"контактный лист позы: {out} ({len(states)} кадров)")
    for name, ev in run_pose_detectors(scene):
        print(f"  {name}: {ev.timecode_sec:.1f} с, confidence {ev.confidence:.2f} — {ev.facts.get('what_visible')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
