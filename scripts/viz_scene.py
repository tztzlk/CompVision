#!/usr/bin/env python3
"""Контактный лист для отладки scene.json: кадры с зонами, кассиром, людьми и значениями состояний.

    PYTHONPATH=. python scripts/viz_scene.py --scene data/scene_cache/Х-20.json --video dataset/videos/Х-20.mov
    PYTHONPATH=. python scripts/viz_scene.py --scene ... --video ... --times 4 5 6 --cols 3

По умолчанию — кадр раз в --every секунд. Результат — JPEG (по умолчанию data/viz/<video_id>.jpg,
папка data/ в .gitignore; кадры датасета в репозиторий не класть). Нужен только OpenCV, модели — нет.
Подписи латиницей: шрифты OpenCV не поддерживают кириллицу.

Легенда: голубой — стол, зелёный — рабочее место, оранжевый — сейф, пурпурный — дверь,
жёлтый — ноутбук; толстая зелёная рамка — кассир, красные — остальные люди (#трек, время появления).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from worker.perception.video import imread_unicode, imwrite_unicode, read_frames_at  # noqa: E402
from worker.scene.contract import SceneState, SceneTimeline, load_scene  # noqa: E402

ZONE_COLORS = {                      # BGR
    "desk": (255, 200, 0),
    "workplace": (0, 200, 0),
    "safe": (0, 140, 255),
    "door": (255, 0, 255),
    "laptop": (0, 255, 255),
}
CASHIER_COLOR = (0, 255, 0)
OTHER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_LABELS = (
    ("safe_open", "safe"), ("cash_on_desk", "cash"), ("bag_on_desk", "bag"), ("outerwear_on_desk", "coat"),
    ("cup_or_food", "cup"), ("phone_in_hand", "phone"), ("laptop_unlocked", "lap"), ("door_open", "door"),
)


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _rect(img: np.ndarray, box, color, thickness: int = 1) -> tuple[int, int]:
    h, w = img.shape[:2]
    p1 = (int(box[0] * w), int(box[1] * h))
    p2 = (int(box[2] * w), int(box[3] * h))
    cv2.rectangle(img, p1, p2, color, thickness)
    return p1


def _text(img: np.ndarray, text: str, org: tuple[int, int], color=TEXT_COLOR, scale: float = 0.4) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw_state(image: np.ndarray, scene: SceneTimeline, state: SceneState, tile_width: int) -> np.ndarray:
    """Кадр, уменьшенный до tile_width, с зонами, людьми и значениями состояний."""
    h, w = image.shape[:2]
    tile = cv2.resize(image, (tile_width, max(1, int(h * tile_width / w))))
    for name, box in scene.zones.items():
        if box is not None:
            corner = _rect(tile, box, ZONE_COLORS.get(name, (200, 200, 200)))
            _text(tile, name, (corner[0] + 2, corner[1] + 12), ZONE_COLORS.get(name, TEXT_COLOR), 0.35)
    if state.cashier_bbox is not None:
        corner = _rect(tile, state.cashier_bbox, CASHIER_COLOR, 3)
        _text(tile, f"cashier p={_fmt(state.cashier_present)}", (corner[0] + 2, max(12, corner[1] - 4)), CASHIER_COLOR)
    for person in state.others:
        corner = _rect(tile, person.bbox, OTHER_COLOR, 2)
        _text(tile, f"#{person.track_id} in@{person.first_seen_t:.1f}", (corner[0] + 2, max(12, corner[1] - 4)), OTHER_COLOR)

    events = " ENTER" if state.person_entered else ""
    events += " EXIT" if state.person_exited else ""
    _text(tile, f"t={state.t:.2f}s ppl={state.persons_in_room} pres={_fmt(state.cashier_present)}{events}",
          (4, 16), OTHER_COLOR if events else TEXT_COLOR, 0.45)
    values = [f"{label} {_fmt(state.value(name))}" for name, label in STATE_LABELS]
    lines = [" ".join(values[:4]), " ".join(values[4:])]
    for i, line in enumerate(lines):
        _text(tile, line, (4, tile.shape[0] - 8 - 14 * (len(lines) - 1 - i)))
    return tile


def render_contact_sheet(
    scene: SceneTimeline,
    frames: Sequence[Optional[np.ndarray]],
    states: Sequence[SceneState],
    *,
    cols: int = 6,
    tile_width: int = 300,
) -> np.ndarray:
    """Склеить плитки в сетку cols столбцов (непрочитанный кадр — серая плитка)."""
    tiles = []
    for image, state in zip(frames, states):
        if image is None:
            height = int(tile_width * scene.frame_size[1] / max(1, scene.frame_size[0]))
            image = np.full((max(1, height), tile_width, 3), 64, dtype=np.uint8)
        tiles.append(draw_state(image, scene, state, tile_width))
    if not tiles:
        raise ValueError("нет кадров для контактного листа")
    tile_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, tile_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def select_states(scene: SceneTimeline, times: Optional[Sequence[float]], every: float) -> list[SceneState]:
    if times:
        return [s for s in (scene.state_at(t) for t in times) if s is not None]
    picked, next_t = [], 0.0
    for state in scene.states:
        if state.t + 1e-6 >= next_t:
            picked.append(state)
            next_t = state.t + every
    return picked


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Контактный лист scene.json для визуальной проверки восприятия.")
    p.add_argument("--scene", type=Path, required=True, help="data/scene_cache/<video_id>.json")
    p.add_argument("--video", type=Path, help="исходное видео (если кадры не сохранены)")
    p.add_argument("--frames-dir", type=Path, help="рабочий каталог с frames/ (build_scene_cache --keep-frames)")
    p.add_argument("--times", type=float, nargs="+", help="конкретные моменты, с")
    p.add_argument("--every", type=float, default=1.0, help="шаг по времени, с (если --times не задан)")
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--tile-width", type=int, default=300)
    p.add_argument("--out", type=Path, help="JPEG (по умолчанию data/viz/<video_id>.jpg)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.video is None and args.frames_dir is None:
        print("нужен --video или --frames-dir", file=sys.stderr)
        return 2
    scene = load_scene(args.scene)
    states = select_states(scene, args.times, args.every)
    if args.frames_dir is not None:
        frames = [imread_unicode(args.frames_dir / s.frame) for s in states]
    else:
        frames = read_frames_at(args.video, [s.t for s in states])
    sheet = render_contact_sheet(scene, frames, states, cols=args.cols, tile_width=args.tile_width)
    out = args.out or ROOT / "data" / "viz" / f"{scene.video_id}.jpg"
    imwrite_unicode(out, sheet, quality=85)
    print(f"контактный лист: {out} ({len(states)} кадров)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
