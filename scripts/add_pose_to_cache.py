#!/usr/bin/env python3
"""Добавить позу в готовый кэш сцен: data/scene_cache/*.json -> data/scene_cache_pose/*.json.

Остальные модели не перезапускаются: кадры вынимаются из видео на моменты состояний
(read_frames_at), ViTPose ставит точки по рамкам из сцены, worker.scene.pose_pass.add_pose
пишет точки и признаки в поля состояний. Запускать на машине с GPU после
python scripts/download_models.py --only vitpose (нужен transformers >= 4.48):

    PYTHONPATH=. python scripts/add_pose_to_cache.py
    PYTHONPATH=. python scripts/add_pose_to_cache.py --only Х-20 --only Х-44
    PYTHONPATH=. python scripts/add_pose_to_cache.py --recompute      # без модели: признаки заново
    PYTHONPATH=. python scripts/add_pose_to_cache.py --recompute --gesture-heads heads/pose_gestures.npz \
        --recompute-to data/scene_cache_pose_heads   # гибрид с головами жестов, без модели

--recompute берёт уже посчитанные точки из --out и пересчитывает только признаки (после правки
PoseConfig) — модель и видео не нужны, работает на CPU за секунды.
Код возврата: 0 — все сцены обработаны; 1 — были ошибки; 2 — сцены не найдены.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.scene.contract import load_scene, save_scene  # noqa: E402
from worker.scene.pose_features import PoseConfig, compute_features  # noqa: E402
from worker.scene.pose_heads import apply_gesture_heads  # noqa: E402
from worker.scene.pose_pass import add_pose, load_heads, pose_frames_from_scene  # noqa: E402

LOG = logging.getLogger("add_pose_to_cache")


def natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem))


def find_scenes(cache: Path, only: Optional[list[str]]) -> list[Path]:
    paths = sorted((p for p in Path(cache).glob("*.json") if not p.name.startswith("_")), key=natural_key)
    if only:
        paths = [p for p in paths if p.stem in set(only)]
    return paths


def find_video(dataset: Path, video_id: str) -> Optional[Path]:
    # os.walk(followlinks=True): каталог видео может быть символической ссылкой (рабочая копия git worktree),
    # а Path.rglob в Python 3.11 по ссылкам на каталоги не ходит.
    for root, _dirs, files in os.walk(dataset, followlinks=True):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() == ".mov" and path.stem == video_id:
                return path
    return None


def recompute(scene_path: Path, out_dir: Path, cfg: PoseConfig, heads: Optional[dict] = None) -> Path:
    """Признаки заново по точкам, уже записанным в сцену (без модели); с heads — и гибрид голов жестов."""
    scene = load_scene(scene_path)
    frames = pose_frames_from_scene(scene)
    features = compute_features(frames, scene.zones, scene.frame_size, cfg)
    if heads:
        features = apply_gesture_heads(frames, features, scene.zones, scene.frame_size, heads)
    states = tuple(
        dataclasses.replace(s, extra={**{k: v for k, v in s.extra.items() if not k.endswith("_head")}, **f})
        for s, f in zip(scene.states, features))
    meta_pose = {k: v for k, v in scene.meta.get("pose", {}).items()
                 if k not in ("gesture_heads", "gesture_heads_oof", "heads_error")}
    if heads:
        meta_pose["gesture_heads"] = {g: h.mode for g, h in sorted(heads.items())}
    return save_scene(dataclasses.replace(scene, states=states, meta={**scene.meta, "pose": meta_pose}),
                      out_dir / scene_path.name)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Поза (ViTPose) в готовый кэш сцен без перезапуска остальных моделей.")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "scene_cache", help="вход: <video_id>.json")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "scene_cache_pose", help="выход: <video_id>.json")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset", help="папка с видео (поиск рекурсивный)")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--every-k", type=int, default=1, help="поза на каждом k-м состоянии")
    p.add_argument("--only", action="append", metavar="VIDEO_ID", help="только это видео (флаг повторяемый)")
    p.add_argument("--recompute", action="store_true", help="только пересчитать признаки по точкам из --out")
    p.add_argument("--recompute-to", type=Path, default=None,
                   help="куда писать пересчитанные сцены (по умолчанию — туда же, в --out)")
    p.add_argument("--gesture-heads", default="",
                   help="головы жестов .npz (относительно --models-dir или абсолютный путь); пусто — только геометрия")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = PoseConfig(every_k=max(1, args.every_k), gesture_heads=args.gesture_heads)
    heads, heads_error = load_heads(cfg, args.models_dir)
    if heads_error:
        LOG.error("%s", heads_error)
        return 2

    if args.recompute:
        scenes = find_scenes(args.out, args.only)
        if not scenes:
            LOG.error("сцены с позой не найдены в %s", args.out)
            return 2
        target = args.recompute_to or args.out
        for path in scenes:
            LOG.info("%s: признаки пересчитаны -> %s", path.stem, recompute(path, target, cfg, heads))
        return 0

    scenes = find_scenes(args.cache, args.only)
    if not scenes:
        LOG.error("сцены не найдены в %s", args.cache)
        return 2

    from worker.perception.pose import load_pose_model
    from worker.perception.video import read_frames_at

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    estimator = load_pose_model(args.models_dir, device)

    failed: list[str] = []
    for i, path in enumerate(scenes, 1):
        vid = path.stem
        started = time.monotonic()
        try:
            scene = load_scene(path)
            video = find_video(args.dataset, scene.video_id)
            if video is None:
                raise FileNotFoundError(f"видео {scene.video_id}.mov не найдено в {args.dataset}")
            frames = read_frames_at(video, [s.t for s in scene.states])
            decoded = time.monotonic() - started
            scene = add_pose(scene, frames, estimator, cfg, heads)
            out = save_scene(scene, args.out / path.name)
        except Exception as exc:  # noqa: BLE001 — одно видео не должно останавливать прогон
            LOG.error("[%d/%d] %s: %s: %s", i, len(scenes), vid, type(exc).__name__, exc)
            failed.append(vid)
            continue
        pose = scene.meta.get("pose", {})
        LOG.info("[%d/%d] %s: кадры %.1f с, поза %.1f с, точки кассира в %s из %d состояний -> %s",
                 i, len(scenes), vid, decoded, pose.get("elapsed_sec", 0.0),
                 pose.get("states_with_cashier_kp"), len(scene.states), out)
    if failed:
        LOG.error("с ошибками: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
