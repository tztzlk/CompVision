#!/usr/bin/env python3
"""Пересчитать зоны и zero-shot SigLIP по сырому кэшу — без RT-DETR и трекинга (треки и предметы COCO
берутся из сырого кэша). Нужен, когда меняется выбор зон или подписи: SigLIP считает по новым вырезкам.

    PYTHONPATH=. python scripts/rezone_raw.py --raw data/raw_cache_v1 --frames data/frames_r2 \
        --out-raw data/raw_cache_r2 [--owl data/owl_r2] [--device cpu|cuda] [--only Х-20]

Кадры (--frames/<video_id>/<номер>.jpg) — те же, что пишет пайплайн с частотой SceneConfig.fps
(для кэша кадров: worker.perception.video.iter_sampled_frames). --owl — каталог с детекциями OWLv2
по ключевым кадрам ({"positions", "owl"}), посчитанными текущим набором запросов ZONE_QUERIES;
без него берутся детекции из сырого кэша (запросы старые). Дальше: scripts/rebuild_from_raw.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.perception import Detection, Frame  # noqa: E402
from worker.scene.builder import FrameRef, SceneConfig  # noqa: E402
from worker.scene.pipeline import _sample_states, check_left_items, scan_desk_items, select_zones  # noqa: E402
from worker.scene.raw_cache import load_raw, rebuild_scene, save_raw  # noqa: E402
from worker.perception.tracking import Track, TrackPoint  # noqa: E402


def _det(d) -> Detection:
    return Detection(label=str(d[0]), score=float(d[1]), bbox_xyxy=tuple(d[2]))  # type: ignore[arg-type]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Зоны и SigLIP заново по сырому кэшу и кэшу кадров")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out-raw", type=Path, required=True)
    ap.add_argument("--owl", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompts", choices=("v1", "v2"), default=None, help="по умолчанию — SceneConfig")
    ap.add_argument("--mask-people", choices=("on", "off"), default=None, help="по умолчанию — SceneConfig")
    ap.add_argument("--only", action="append")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from worker.perception.models import load_models

    bundle = load_models(args.models_dir, args.device)
    over = {}
    if args.prompts:
        over["state_prompts"] = args.prompts
    if args.mask_people:
        over["mask_people_in_desk"] = args.mask_people == "on"
    cfg = dataclasses.replace(SceneConfig(), **over)
    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    for path in files:
        vid = path.stem
        if args.only and vid not in args.only:
            continue
        started = time.monotonic()
        raw = load_raw(path)
        refs = [FrameRef(index=int(i), t=float(t), frame=str(f)) for i, t, f in raw["frames"]]
        frames = [Frame(index=r.index, timestamp_sec=r.t, path=args.frames / vid / Path(r.frame).name) for r in refs]
        tracks = [Track(int(t["id"]), [TrackPoint(int(fi), float(tt), tuple(b), float(s))  # type: ignore[arg-type]
                                       for fi, tt, b, s in t["points"]]) for t in raw["tracks"]]
        objects = [[_det(d) for d in dets] for dets in raw["objects"]]
        owl_file = args.owl / f"{vid}.json" if args.owl else None
        if owl_file is not None and owl_file.is_file():
            owl = [[_det(d) for d in dets] for dets in json.loads(owl_file.read_text(encoding="utf-8"))["owl"]]
        else:
            owl = [[_det(d) for d in dets] for dets in raw.get("owl_keyframes", [])]
        zones, info = select_zones(owl, objects, tracks, cfg)
        persons: list[list[Detection]] = [[] for _ in frames]
        for tr in tracks:
            for p in tr.points:
                if 0 <= p.frame_index < len(persons):
                    persons[p.frame_index].append(Detection("person", p.score, p.bbox))
        samples = _sample_states(frames, zones, bundle, cfg, persons)
        meta = {**raw.get("meta", {}), "zones": info, "rezoned": True}
        out = args.out_raw / f"{vid}.json"
        common = dict(video_id=raw["video_id"], duration_sec=raw["duration_sec"],
                      frame_size=tuple(raw["frame_size"]), frames=refs, tracks=tracks, objects=objects, zones=zones,
                      zone_info=info, owl_keyframes=owl, state_samples=samples, config=cfg, meta=meta,
                      brightness=raw.get("brightness"))
        item_scan = scan_desk_items(frames, bundle, cfg)
        save_raw(out, **common, item_scan=item_scan)
        # предметы на столе после уходов кассира — по сцене, собранной текущим builder
        checks = check_left_items(rebuild_scene(load_raw(out), cfg), frames, bundle, cfg)
        if checks:
            save_raw(out, **common, item_checks=checks, item_scan=item_scan)
        z = {k: [round(v, 2) for v in b] if b else None for k, b in zones.items() if k in ("desk", "safe")}
        print(f"{vid}: {time.monotonic() - started:.0f} с, {z}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
