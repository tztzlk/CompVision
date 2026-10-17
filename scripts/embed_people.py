#!/usr/bin/env python3
"""Эмбеддинги SigLIP вырезок людей по трекам сырого кэша + геометрия — данные головы «внутри кассы».

    PYTHONPATH=. .venv/bin/python scripts/embed_people.py sheet --only Х-33     # лист треков для разметки
    PYTHONPATH=. .venv/bin/python scripts/embed_people.py embed                  # data/emb/person/<вариант>/<vid>.npz

Треки — data/raw_cache_r3/<video_id>.json (tracks: id, points [кадр, t, рамка, score]), кадры —
data/frames_r2/<video_id>/. Точка трека берётся раз в --every с. Вырезка — рамка человека с полем
worker.perception.heads.PERSON_MARGIN. Геометрия (GEO_NAMES): высота рамки к типичной высоте человека в кассе (track_height),
низ рамки относительно верхнего края стола, score детекции, центр рамки.

sheet -> data/ml_results/r3/sheets/person/<video_id>.jpg: строка на трек, кадры с его рамкой (красная)
и рамкой стола (зелёная), подпись «id t».
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from augment_videos import VARIANTS, make_transform, transform_bbox, variant_params  # noqa: E402
from worker.perception.heads import crop_person  # noqa: E402
from worker.perception.tracking import Track, TrackPoint  # noqa: E402
from worker.perception.video import imread_unicode  # noqa: E402
from worker.scene.builder import track_height  # noqa: E402

GEO_NAMES = ("height_ratio", "bottom_minus_desk_top", "score", "cx", "cy", "bottom")
ALL_VARIANTS = ("orig", *VARIANTS)


def raw_tracks(raw: dict[str, Any]) -> list[Track]:
    return [Track(int(t["id"]), [TrackPoint(int(fi), float(tt), tuple(b), float(s)) for fi, tt, b, s in t["points"]])
            for t in raw["tracks"]]


def sample_points(track: dict[str, Any], every: float) -> list[tuple[int, float, list[float], float]]:
    out, next_t = [], -1.0
    for fi, t, b, s in track["points"]:
        if t + 1e-6 >= next_t:
            out.append((int(fi), float(t), list(b), float(s)))
            next_t = t + every
    return out


def geo(bbox: Any, score: float, ref_h: float, desk: Optional[Any]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [(y2 - y1) / ref_h if ref_h > 0 else 0.0,
            (y2 - desk[1]) if desk is not None else 0.0,
            score, (x1 + x2) / 2, (y1 + y2) / 2, y2]


def iter_people(raw: dict[str, Any], frames_dir: Path, every: float, variant: str, work_layout: bool = False
                ) -> Iterator[tuple[int, float, Any, list[float]]]:
    """(track_id, t, вырезка, геометрия) — геометрия всегда по исходным рамкам (как видит builder).

    work_layout — кадры в рабочем каталоге build_scene_cache.py --keep-frames, frames_dir — сам каталог."""
    w, h = raw["frame_size"]
    ref_h = track_height(raw_tracks(raw))
    desk = raw["zones"].get("desk")
    fn, params = None, None
    if variant != "orig":
        params = variant_params(variant)
        fn = make_transform(variant, params, w, h)
    cache: dict[int, Any] = {}
    rel = {int(i): r for i, _t, r in raw["frames"]}
    if work_layout:
        from build_scene_cache import safe_dirname

        base = frames_dir / safe_dirname(raw["video_id"])
    for tr in raw["tracks"]:
        for fi, t, b, s in sample_points(tr, every):
            if fi not in cache:
                img = imread_unicode(base / rel[fi] if work_layout else frames_dir / f"{fi:06d}.jpg")
                cache[fi] = fn(img) if (img is not None and fn is not None) else img
            img = cache[fi]
            if img is None:
                continue
            box = transform_bbox(variant, params, b, w, h) if fn is not None else b
            crop = crop_person(img, box)
            if crop is not None:
                yield int(tr["id"]), t, crop, geo(b, s, ref_h, desk)


def load_raw(raw_dir: Path, only: Optional[list[str]]) -> list[dict[str, Any]]:
    out = []
    for path in sorted(raw_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if only and raw["video_id"] not in only:
            continue
        out.append(raw)
    return out


def cmd_sheet(args: argparse.Namespace) -> int:
    out_dir = args.sheets / "person"
    out_dir.mkdir(parents=True, exist_ok=True)
    tw, th = 128, 234
    for raw in load_raw(args.raw, args.only):
        vid = raw["video_id"]
        desk = raw["zones"].get("desk")
        rows = []
        for tr in raw["tracks"]:
            pts = sample_points(tr, args.every)
            if len(pts) > args.cols:
                idx = np.linspace(0, len(pts) - 1, args.cols).round().astype(int)
                pts = [pts[i] for i in idx]
            tiles = []
            for fi, t, b, s in pts:
                img = imread_unicode(args.frames / vid / f"{fi:06d}.jpg")
                if img is None:
                    continue
                img = img.copy()
                H, W = img.shape[:2]
                if desk is not None:
                    cv2.rectangle(img, (int(desk[0] * W), int(desk[1] * H)), (int(desk[2] * W), int(desk[3] * H)), (0, 200, 0), 2)
                cv2.rectangle(img, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)), (0, 0, 255), 3)
                tile = cv2.resize(img, (tw, th))
                tile = cv2.copyMakeBorder(tile, 0, 16, 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(tile, f"#{tr['id']} {t:.0f}s {s:.2f}", (2, th + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
                tiles.append(tile)
            if tiles:
                blank = np.full_like(tiles[0], 255)
                rows.append(np.hstack(tiles + [blank] * (args.cols - len(tiles))))
        if not rows:
            continue
        ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        (out_dir / f"{vid}.jpg").write_bytes(buf.tobytes())
        print(out_dir / f"{vid}.jpg")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoProcessor, SiglipModel

    torch.set_num_threads(args.threads)
    path = ROOT / "models" / "hf" / "siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = SiglipModel.from_pretrained(path, local_files_only=True).to(args.device).eval()
    raws = load_raw(args.raw, args.only)
    for variant in args.variant or ALL_VARIANTS:
        out_dir = args.out / "person" / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        t0, n = time.time(), 0
        for raw in raws:
            target = out_dir / f"{raw['video_id']}.npz"
            if target.exists() and not args.force:
                continue
            frames_dir = args.frames if args.work_layout else args.frames / raw["video_id"]
            items = list(iter_people(raw, frames_dir, args.every, variant, args.work_layout))
            if not items:
                continue
            embs = []
            with torch.inference_mode():
                for s in range(0, len(items), 32):
                    chunk = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, _, c, _ in items[s:s + 32]]
                    px = processor(images=chunk, return_tensors="pt")["pixel_values"].to(args.device)
                    e = torch.nn.functional.normalize(model.get_image_features(pixel_values=px).float(), dim=-1)
                    embs.append(e.cpu().numpy())
            np.savez_compressed(target, track=np.array([i[0] for i in items], dtype=np.int32),
                                t=np.array([i[1] for i in items], dtype=np.float32),
                                emb=np.concatenate(embs).astype(np.float32),
                                geo=np.array([i[3] for i in items], dtype=np.float32))
            n += len(items)
        print(f"[{variant}] {n} вырезок людей за {time.time() - t0:.0f} с -> {out_dir}", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Вырезки людей по трекам: эмбеддинги и листы разметки.")
    p.add_argument("cmd", choices=("embed", "sheet"))
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw_cache_r3")
    p.add_argument("--frames", type=Path, default=ROOT / "data" / "frames_r2")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "emb")
    p.add_argument("--sheets", type=Path, default=ROOT / "data" / "ml_results" / "r3" / "sheets")
    p.add_argument("--variant", action="append", choices=ALL_VARIANTS)
    p.add_argument("--only", action="append")
    p.add_argument("--every", type=float, default=1.0)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--work-layout", action="store_true",
                   help="--frames — рабочий каталог build_scene_cache.py --keep-frames (кэши аугментаций)")
    args = p.parse_args(argv)
    return cmd_embed(args) if args.cmd == "embed" else cmd_sheet(args)


if __name__ == "__main__":
    sys.exit(main())
