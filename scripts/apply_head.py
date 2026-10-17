#!/usr/bin/env python3
"""Подставить голову «сейф открыт» в сырой кэш без GPU: safe_open := вероятность головы.

    PYTHONPATH=. .venv/bin/python scripts/apply_head.py --raw data/raw_cache_r3 --emb data/emb/safe/orig \
        --out data/raw_cache_r3_head
    PYTHONPATH=. .venv/bin/python scripts/rebuild_from_raw.py --raw data/raw_cache_r3_head --out data/scene_cache_r3_head --code-config

Эмбеддинги — scripts/embed_zones.py (та же зона, те же секунды, что state_samples сырого кэша).
По умолчанию вероятности ВНЕ ФОЛДА: для видео — модель фолда, в котором его группа не училась
(data/ml_results/r3/heads/safe_open_folds.npz из scripts/train_heads.py) — честная оценка на 40 видео.
--final — финальная голова models/heads/safe_open.npz (то, что увидит сервис).
Прежнее zero-shot-значение остаётся в state_samples.safe_open_zero_shot; meta.safe_head и
config.safe_head отмечают, что сцена считалась с головой (детектор R10 берёт head_baseline_sec).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cache import known_aliases, normalize_video_id  # noqa: E402
from train_heads import GEO_SCALE, group_of  # noqa: E402


def person_geo_lookup_one(raw: dict) -> dict[tuple[int, int], list[float]]:
    """(track_id, t в мс) -> heads.person_geo для одного сырого кэша (t из .npz — float32)."""
    from worker.perception.heads import person_geo
    from worker.perception.tracking import Track, TrackPoint
    from worker.scene.builder import track_height

    tracks = [Track(int(t["id"]), [TrackPoint(int(a), float(b), tuple(c), float(d)) for a, b, c, d in t["points"]])
              for t in raw["tracks"]]
    ref, desk = track_height(tracks), raw["zones"].get("desk")
    return {(int(t["id"]), int(round(float(tt) * 1000))): person_geo(box, sc, ref, desk)
            for t in raw["tracks"] for _fi, tt, box, sc in t["points"]}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="safe_open из головы в сыром кэше.")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--emb", type=Path, default=None, help="каталог <video_id>.npz эмбеддингов вырезок сейфа")
    p.add_argument("--person-emb", type=Path, default=None,
                   help="каталог эмбеддингов людей (scripts/embed_people.py) — добавить person_head")
    p.add_argument("--person-folds", type=Path,
                   default=ROOT / "data" / "ml_results" / "r3" / "heads" / "person_inside_folds.npz")
    p.add_argument("--person-head", type=Path, default=ROOT / "models" / "heads" / "person_inside.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=Path, default=ROOT / "data" / "ml_results" / "r3" / "heads" / "safe_open_folds.npz")
    p.add_argument("--final", action="store_true", help="финальная голова вместо моделей фолдов")
    p.add_argument("--head", type=Path, default=ROOT / "models" / "heads" / "safe_open.npz")
    args = p.parse_args(argv)

    aliases = {normalize_video_id(k): v for k, v in known_aliases().items()}
    pmodels = None
    if args.person_emb is not None:
        if args.final:
            h = np.load(args.person_head)
            pmodels = (h["w"][None], np.array([float(h["b"])]), None)
        else:
            f = np.load(args.person_folds)
            pmodels = (f["w"], f["b"], json.loads(str(f["groups"])))
    if args.emb is None:
        models = None
    elif args.final:
        h = np.load(args.head)
        models = None
        w_final, b_final = h["w"], float(h["b"])
    else:
        f = np.load(args.folds)
        models = (f["w"], f["b"], json.loads(str(f["groups"])))
    args.out.mkdir(parents=True, exist_ok=True)
    n_vid = n_rep = 0
    for path in sorted(args.raw.glob("*.json")):
        if path.name.startswith("_"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        vid = normalize_video_id(raw["video_id"])
        emb_path = (args.emb / f"{raw['video_id']}.npz") if args.emb is not None else None
        samples = raw["state_samples"]
        if pmodels is not None and (args.person_emb / f"{raw['video_id']}.npz").exists():
            pdata = np.load(args.person_emb / f"{raw['video_id']}.npz")
            fold = 0 if pmodels[2] is None else (pmodels[2].get(group_of(vid, aliases)) or 0)
            geo_by = person_geo_lookup_one(raw)
            points = {(int(t["id"]), int(round(float(pt[1]) * 1000))): pt for t in raw["tracks"] for pt in t["points"]}
            head_rows = []
            for tr, t, e in zip(pdata["track"], pdata["t"], pdata["emb"]):
                key = (int(tr), int(round(float(t) * 1000)))
                pt, g = points.get(key), geo_by.get(key)
                if pt is None or g is None:
                    continue
                x = np.concatenate([e, np.asarray(g, dtype=np.float32) * np.asarray(GEO_SCALE, dtype=np.float32)])
                pr = 1.0 / (1.0 + np.exp(-(x @ pmodels[0][fold] + float(pmodels[1][fold]))))
                head_rows.append([int(pt[0]), pt[2], round(float(pr), 4)])
            raw["person_head"] = head_rows
            raw["meta"] = {**(raw.get("meta") or {}), "inside_head": {"name": "person_inside", "oof": not args.final,
                                                                       "points": len(head_rows)}}
        if args.emb is not None and (args.emb / f"{raw['video_id']}.npz").exists() and samples.get("safe_open"):
            data = np.load(emb_path)
            if models is None:
                w, b = w_final, b_final
            else:
                fold = models[2].get(group_of(vid, aliases))
                if fold is None:          # группа не встречалась при обучении — любая модель честна
                    fold = 0
                w, b = models[0][fold], float(models[1][fold])
            probs = 1.0 / (1.0 + np.exp(-(data["emb"] @ w + b)))
            by_t = {round(float(t), 1): float(pr) for t, pr in zip(data["t"], probs)}
            zs = samples["safe_open"]
            new = []
            for t, pz in zs:
                ph = by_t.get(round(float(t), 1))
                new.append([t, round(ph, 4) if ph is not None else None])
                n_rep += ph is not None
            samples["safe_open_zero_shot"] = zs
            samples["safe_open"] = new
            raw["meta"] = {**(raw.get("meta") or {}), "safe_head": {"name": "safe_open", "oof": not args.final}}
            raw["config"] = {**(raw.get("config") or {}), "safe_head": "heads/safe_open.npz"}
            n_vid += 1
        (args.out / path.name).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    print(f"голова подставлена в {n_vid} видео ({n_rep} выборок) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
