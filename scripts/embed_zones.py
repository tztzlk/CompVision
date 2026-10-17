#!/usr/bin/env python3
"""Эмбеддинги SigLIP вырезок зоны (сейф) по кадрам датасета — данные для голов (scripts/train_heads.py).

    PYTHONPATH=. .venv/bin/python scripts/embed_zones.py embed --zone safe            # оригинал + все аугментации
    PYTHONPATH=. .venv/bin/python scripts/embed_zones.py sheet --zone safe --only Х-20  # контактный лист для разметки

Кадры — data/frames_r2/<video_id>/ (3 к/с), зона — из сырого кэша data/raw_cache_r3/<video_id>.json,
вырезка — worker.perception.states.crop_zone с полем ZONE_CROP_MARGIN[zone], раз в --every с.
Аугментации (scripts/augment_videos.py) применяются к тем же кадрам, рамка зоны пересчитывается тем же
преобразованием — метка оригинала подходит к вырезке как есть. Видеокарта не нужна (по умолчанию CPU).

embed -> data/emb/<zone>/<вариант>/<video_id>.npz: t (с), emb (N×768, нормированные), bbox (рамка зоны).
sheet -> data/ml_results/r3/sheets/<zone>/<video_id>.jpg: вырезки с подписью t.
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
from worker.perception.states import ZONE_CROP_MARGIN, crop_zone  # noqa: E402
from worker.perception.video import imread_unicode  # noqa: E402

ALL_VARIANTS = ("orig", *VARIANTS)


def iter_crops(raw: dict[str, Any], frames_dir: Path, zone: str, every: float, variant: str,
               work_layout: bool = False) -> Iterator[tuple[float, Any, tuple[float, ...]]]:
    """(t, вырезка BGR, рамка зоны) раз в every с; пусто, если зоны нет.

    work_layout — кадры в рабочем каталоге build_scene_cache.py --keep-frames (<work>/<ascii-имя>/frames/…),
    frames_dir — сам <work>; иначе frames_dir/<номер>.jpg (data/frames_r2/<video_id>/)."""
    bbox = raw["zones"].get(zone)
    if bbox is None:
        return
    w, h = raw["frame_size"]
    fn = None
    if variant != "orig":
        params = variant_params(variant)
        fn = make_transform(variant, params, w, h)
        bbox = transform_bbox(variant, params, bbox, w, h)
    if work_layout:
        from build_scene_cache import safe_dirname

        base = frames_dir / safe_dirname(raw["video_id"])
    next_t = 0.0
    for idx, t, rel in raw["frames"]:
        if t + 1e-6 < next_t:
            continue
        image = imread_unicode(base / rel if work_layout else frames_dir / f"{int(idx):06d}.jpg")
        if image is None:
            continue
        next_t = t + every
        if fn is not None:
            image = fn(image)
        crop = crop_zone(image, tuple(bbox), ZONE_CROP_MARGIN.get(zone, 0.1))
        if crop is not None:
            yield float(t), crop, tuple(bbox)


def load_raw(raw_dir: Path, only: Optional[list[str]]) -> list[dict[str, Any]]:
    out = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if only and raw["video_id"] not in only:
            continue
        out.append(raw)
    return out


def cmd_embed(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoProcessor, SiglipModel

    torch.set_num_threads(args.threads)
    path = ROOT / "models" / "hf" / "siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = SiglipModel.from_pretrained(path, local_files_only=True).to(args.device).eval()
    raws = load_raw(args.raw, args.only)
    for variant in args.variant or ALL_VARIANTS:
        out_dir = args.out / args.zone / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        t0, n_total = time.time(), 0
        for raw in raws:
            vid = raw["video_id"]
            target = out_dir / f"{vid}.npz"
            if target.exists() and not args.force:
                continue
            frames_dir = args.frames if args.work_layout else args.frames / vid
            items = list(iter_crops(raw, frames_dir, args.zone, args.every, variant, args.work_layout))
            if not items:
                continue
            embs = []
            with torch.inference_mode():
                for s in range(0, len(items), 32):
                    chunk = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, c, _ in items[s:s + 32]]
                    px = processor(images=chunk, return_tensors="pt")["pixel_values"].to(args.device)
                    e = torch.nn.functional.normalize(model.get_image_features(pixel_values=px).float(), dim=-1)
                    embs.append(e.cpu().numpy())
            np.savez_compressed(target, t=np.array([t for t, _, _ in items], dtype=np.float32),
                                emb=np.concatenate(embs).astype(np.float32),
                                bbox=np.array(items[0][2], dtype=np.float32))
            n_total += len(items)
        print(f"[{variant}] {n_total} вырезок за {time.time() - t0:.0f} с -> {out_dir}", flush=True)
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    out_dir = args.sheets / args.zone
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw in load_raw(args.raw, args.only):
        items = list(iter_crops(raw, args.frames / raw["video_id"], args.zone, args.every, "orig"))
        if not items:
            continue
        th = args.tile
        tiles = []
        for t, crop, _ in items:
            tw = max(1, int(round(crop.shape[1] * th / crop.shape[0])))
            tile = cv2.resize(crop, (tw, th))
            tile = cv2.copyMakeBorder(tile, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, f"{t:.0f}", (2, th + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tiles.append(tile)
        tw = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, tw - t.shape[1] + 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for t in tiles]
        cols = args.cols
        rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
        blank = np.full_like(tiles[0], 255)
        grid = np.vstack([np.hstack(r + [blank] * (cols - len(r))) for r in rows])
        target = out_dir / f"{raw['video_id']}.jpg"
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        target.write_bytes(buf.tobytes())
        print(target)
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Эмбеддинги SigLIP вырезок зоны и контактные листы.")
    p.add_argument("cmd", choices=("embed", "sheet"))
    p.add_argument("--zone", default="safe")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw_cache_r3")
    p.add_argument("--frames", type=Path, default=ROOT / "data" / "frames_r2")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "emb")
    p.add_argument("--sheets", type=Path, default=ROOT / "data" / "ml_results" / "r3" / "sheets")
    p.add_argument("--variant", action="append", choices=ALL_VARIANTS, help="по умолчанию — оригинал и все аугментации")
    p.add_argument("--only", action="append", help="video_id, можно несколько")
    p.add_argument("--every", type=float, default=1.0, help="шаг по времени, с")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--work-layout", action="store_true",
                   help="--frames — рабочий каталог build_scene_cache.py --keep-frames (кэши аугментаций)")
    p.add_argument("--tile", type=int, default=110, help="высота плитки листа, px")
    p.add_argument("--cols", type=int, default=12)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    return cmd_embed(args) if args.cmd == "embed" else cmd_sheet(args)


if __name__ == "__main__":
    sys.exit(main())
