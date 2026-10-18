#!/usr/bin/env python3
"""Зона двери по сигналу прохода (SceneConfig.door_pick_signal) в готовый сырой кэш — без моделей и GPU.

    PYTHONPATH=. python scripts/redoor_raw.py --raw data/raw_cache_r3 --frames data/frames_r3 \
        --out data/raw_cache_r3_door [--off] [--only Х-32]

По сохранённым детекциям OWLv2 (owl_keyframes), трекам и кадрам (--frames/<video_id>/<номер>.jpg):
  1. дверь — worker.perception.zones.infer_zones с выбором worker.scene.pipeline.door_selector (--off —
     без него, лучшая группа по сумме score: база для сравнения на той же пересборке);
  2. сейф — заново (pick_safe отбрасывает кандидатов в зоне двери). Сменился — meta.redoor.safe_changed:
     zero-shot и дообученная башня сейфа в state_samples посчитаны по старой вырезке и верны только после
     пересборки на GPU;
  3. door_diff — для новой двери, как scripts/add_door_to_raw.py;
  4. config.door_pick_signal — чем выбрана дверь: rebuild_from_raw.py --rezone (и с --code-config) по нему
     сохраняет её, а не берёт лучшую по сумме score с door_diff другой рамки.
Люди (закрашиваются в полосе двери и подтверждают проход) — точки треков, как в add_door_to_raw.
Стол и ноутбук берутся из кэша. Остальное в файле не меняется. Дальше — scripts/rebuild_from_raw.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.perception import Detection, Frame  # noqa: E402
from worker.perception.geometry import iou  # noqa: E402
from worker.perception.tracking import Track, TrackPoint  # noqa: E402
from worker.scene.builder import SceneConfig  # noqa: E402
from worker.scene.pipeline import door_selector, door_signal, select_zones  # noqa: E402
from worker.scene.raw_cache import load_raw  # noqa: E402

# поля SceneConfig, которыми собрано восприятие зон (берутся из кэша, остальное — из кода)
ZONE_FIELDS = ("fps", "desk_by_people", "zone_keyframes", "zone_min_score")
# детекции в кэше округлены до 4 знаков: пересчитанная рамка той же группы отличается на 1e-4, а если рамка
# на границе IoU 0,5 группировки — заметнее (IoU 0,98: Х-16 flip, Х-27 persp). При IoU не меньше SAME зона
# считается прежней и берётся из кэша как есть (вместе с door_diff)
SAME = 0.9


def _det(d: Any) -> Detection:
    return Detection(label=str(d[0]), score=float(d[1]), bbox_xyxy=tuple(d[2]))  # type: ignore[arg-type]


def _box(b: Any) -> Optional[list[float]]:
    return None if b is None else [round(float(v), 4) for v in b]


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return iou(tuple(a), tuple(b)) >= SAME


def redoor(raw: dict[str, Any], frames_dir: Path, cfg: SceneConfig) -> dict[str, Any]:
    """Новый сырой кэш одного видео: дверь, сейф, door_diff и meta.redoor (остальное — как было)."""
    frames = [Frame(index=int(i), timestamp_sec=float(t), path=frames_dir / Path(f).name) for i, t, f in raw["frames"]]
    tracks = [Track(int(t["id"]), [TrackPoint(int(fi), float(tt), tuple(b), float(s))  # type: ignore[arg-type]
                                   for fi, tt, b, s in t["points"]]) for t in raw["tracks"]]
    objects = [[_det(d) for d in dets] for dets in raw["objects"]]
    owl = [[_det(d) for d in dets] for dets in raw.get("owl_keyframes", [])]
    persons: list[list[Detection]] = [[] for _ in frames]
    for tr in tracks:
        for p in tr.points:
            if 0 <= p.frame_index < len(persons):
                persons[p.frame_index].append(Detection("person", p.score, p.bbox))
    door_select, cache = door_selector(frames, persons, cfg) if cfg.door_pick_signal else (None, {})
    zones, info = select_zones(owl, objects, tracks, cfg, door_select)

    old = raw.get("zones") or {}
    out = dict(raw)
    new_zones = dict(old)
    new_info = dict(raw.get("zone_info") or {})
    changed = {}
    for name in ("door", "safe"):
        changed[name] = not _same(old.get(name), zones.get(name))
        if changed[name]:
            new_zones[name] = _box(zones.get(name))
        new_info[name] = {**(new_info.get(name) or {}), **(info.get(name) or {})} if not changed[name] else info.get(name)
    door = zones.get("door")
    if cfg.door_change and (changed["door"] or raw.get("door_diff") is None):
        diffs = cache[tuple(door)] if door is not None and tuple(door) in cache else door_signal(frames, persons, door, cfg)
        out["door_diff"] = None if diffs is None else [None if d is None else round(float(d), 4) for d in diffs]
    out["zones"], out["zone_info"] = new_zones, new_info
    # чем выбрана дверь: rebuild_scene(rezone=True) по нему сохраняет её (_keep_door), иначе берёт лучшую
    # по сумме score, и зона расходится с door_diff
    out["config"] = {**(raw.get("config") or {}), "door_pick_signal": cfg.door_pick_signal}
    door_changed, safe_changed = changed["door"], changed["safe"]
    meta = dict(raw.get("meta") or {})
    meta["zones"] = new_info
    meta["redoor"] = {
        "signal": cfg.door_pick_signal,
        "door_changed": door_changed, "old_door": _box(old.get("door")),
        "safe_changed": safe_changed, "old_safe": _box(old.get("safe")),
        # zero-shot / башня сейфа считались по вырезке старой зоны сейфа
        "safe_samples_stale": safe_changed,
        # стол и ноутбук из кэша; расхождение с пересчётом — кэш собран другой версией выбора зон
        "kept_mismatch": [k for k in ("desk", "laptop") if not _same(old.get(k), zones.get(k))],
    }
    out["meta"] = meta
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Зона двери по сигналу прохода в готовый сырой кэш, без моделей")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--off", action="store_true", help="без сигнала (door_pick_signal=false): база для сравнения")
    ap.add_argument("--only", action="append")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if args.out.resolve() == args.raw.resolve():
        ap.error("--out должен отличаться от --raw")
    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    done = door_n = safe_n = 0
    for path in files:
        vid = path.stem
        if args.only and vid not in args.only:
            continue
        raw = load_raw(path)
        keep = {k: raw["config"][k] for k in ZONE_FIELDS if k in raw.get("config", {})}
        # сигнал включён независимо от умолчания SceneConfig.door_pick_signal (скрипт ради него), --off — без него
        cfg = dataclasses.replace(SceneConfig(), **keep, door_pick_signal=not args.off)
        out = redoor(raw, args.frames / vid, cfg)
        dst = args.out / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dst)
        r = out["meta"]["redoor"]
        sig = (out["zone_info"].get("door") or {}).get("signal") or {}
        door_n += r["door_changed"]
        safe_n += r["safe_changed"]
        print(f"{vid}: дверь {'сменилась' if r['door_changed'] else 'та же'}"
              f"{', сейф сменился' if r['safe_changed'] else ''}"
              f"{', стол/ноутбук не совпали: ' + ','.join(r['kept_mismatch']) if r['kept_mismatch'] else ''}"
              f"  act {sig.get('act')}", flush=True)
        done += 1
    print(f"{done} файлов -> {args.out}: дверь сменилась на {door_n}, сейф на {safe_n} "
          f"(zero-shot сейфа там устарел)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
