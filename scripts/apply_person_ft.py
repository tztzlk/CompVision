#!/usr/bin/env python3
"""Вероятности башни «человек внутри кассы» в сырой кэш без GPU: поле person_ft = [[кадр, рамка, p], ...].

    PYTHONPATH=. .venv/bin/python scripts/apply_person_ft.py --raw data/ml_results/dep/raw_base \\
        --probs data/ml_results/inside_ft/cv_b4_geo/person_inside_ft_oof.npz --out data/ml_results/inside_ft/raw_ft
    PYTHONPATH=. .venv/bin/python scripts/rebuild_from_raw.py --raw data/ml_results/inside_ft/raw_ft \\
        --out data/ml_results/inside_ft/scene_ft --code-config --set inside_ft=ft/person_inside

--probs — вывод scripts/finetune_siglip.py person-cv (вне фолда: для видео — модель фолда, где его группа
не училась) или person-final (person_inside_ft_final.npz, не вне фолда). Точки сопоставляются по (трек, кадр);
рамка берётся из сырого кэша — тот же ключ builder._head_key, что у точек треков. meta.inside_ft отмечает,
откуда вероятности. Без --set inside_ft=... builder поле не использует.
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

from eval_cache import normalize_video_id  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="person_ft из вероятностей башни в сыром кэше.")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--probs", type=Path, required=True, help="npz: video, track, frame, p (finetune_siglip.py person-*)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = np.load(args.probs)
    by: dict[tuple[str, int, int], float] = {}
    for v, tr, fi, p in zip(d["video"], d["track"], d["frame"], d["p"]):
        if not np.isnan(p):
            by[(normalize_video_id(str(v)), int(tr), int(fi))] = float(p)
    args.out.mkdir(parents=True, exist_ok=True)
    n_vid = n_pts = n_miss = 0
    for path in sorted(args.raw.glob("*.json")):
        if path.name.startswith("_"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        vid = normalize_video_id(raw["video_id"])
        rows = []
        for t in raw["tracks"]:
            for fi, _t, box, _sc in t["points"]:
                p = by.get((vid, int(t["id"]), int(fi)))
                if p is None:
                    n_miss += 1
                    continue
                rows.append([int(fi), box, round(p, 4)])
        raw["person_ft"] = rows
        raw["meta"] = {**(raw.get("meta") or {}), "inside_ft": {"probs": str(args.probs), "points": len(rows)}}
        (args.out / path.name).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        n_vid, n_pts = n_vid + 1, n_pts + len(rows)
    print(f"{n_vid} видео, {n_pts} точек с p, без p {n_miss} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
