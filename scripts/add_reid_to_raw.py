#!/usr/bin/env python3
"""Добавить в сырой кэш эмбеддинги треков для повторной идентификации кассира (SceneConfig.cashier_reid).

    PYTHONPATH=. .venv/bin/python scripts/add_reid_to_raw.py --raw data/ml_results/dep/raw_base \\
        --emb data/emb/person/orig --out data/ml_results/inside_ft/reid/raw_reid

Эмбеддинги вырезок людей — scripts/embed_people.py (data/emb/person/<вариант>/<video_id>.npz: track, t, emb;
точка трека раз в --every с, треки — те же, что в сыром кэше). На трек — средний нормированный эмбеддинг
первых и последних --points точек: raw["person_reid"] = [[id, first, last], ...] (округление до 4 знаков).
Треки без эмбеддинга пропускаются. Остальной сырой кэш копируется без изменений.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def track_embeddings(npz: Path, points: int) -> dict[int, tuple[list[float], list[float]]]:
    """{id трека: (эмбеддинг первых points точек, последних)} — средние нормированных, снова нормированные."""
    d = np.load(npz)
    tracks, times, emb = d["track"], d["t"], _unit(d["emb"].astype(np.float64))
    out: dict[int, tuple[list[float], list[float]]] = {}
    for tid in np.unique(tracks):
        idx = np.where(tracks == tid)[0]
        idx = idx[np.argsort(times[idx], kind="stable")]
        first = _unit(emb[idx[:points]].mean(0))
        last = _unit(emb[idx[-points:]].mean(0))
        out[int(tid)] = ([round(float(x), 4) for x in first], [round(float(x), 4) for x in last])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="person_reid в сырой кэш по эмбеддингам scripts/embed_people.py")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data" / "emb" / "person" / "orig")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", type=int, default=2, help="сколько первых / последних точек трека усреднять")
    args = ap.parse_args(argv)
    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    args.out.mkdir(parents=True, exist_ok=True)
    missing = 0
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        npz = args.emb / f"{path.stem}.npz"
        if npz.exists():
            embs = track_embeddings(npz, args.points)
            raw["person_reid"] = [[t["id"], *embs[t["id"]]] for t in raw["tracks"] if t["id"] in embs]
        else:
            missing += 1
            print(f"{path.stem}: нет {npz}", file=sys.stderr)
        tmp = args.out / (path.name + ".tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, args.out / path.name)
    print(f"person_reid: {len(files) - missing} из {len(files)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
