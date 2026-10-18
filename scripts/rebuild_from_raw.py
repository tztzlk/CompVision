#!/usr/bin/env python3
"""Пересобрать кэш сцен из сырого кэша текущим builder — без GPU и моделей (секунды на видео).

    PYTHONPATH=. python scripts/rebuild_from_raw.py [--raw data/raw_cache] [--out data/scene_cache_rebuilt]

Сырой кэш пишет scripts/build_scene_cache.py --raw-out data/raw_cache на машине с GPU.
Пороги SceneConfig берутся из сырого кэша (как при сборке) — кроме тех, что новые в коде:
у них значения по умолчанию. --set имя=значение (повторяемо) — поле SceneConfig поверх этого. Дальше замер: scripts/eval_cache.py --cache <out>.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.scene import save_scene  # noqa: E402
from worker.scene.builder import SceneConfig  # noqa: E402
from worker.scene.raw_cache import load_raw, rebuild_scene  # noqa: E402


# Поля SceneConfig, которыми собрано восприятие (кадры, зоны, zero-shot): с --code-config берутся из кэша.
# door_pick_signal — дверь выбрана по сигналу прохода (door_diff в кэше посчитан для неё): без него --rezone
# выбрал бы группу OWLv2 по сумме score, и зона двери разошлась бы с door_diff.
PERCEPTION_FIELDS = ("fps", "state_prompts", "mask_people_in_desk", "desk_by_people", "zone_keyframes",
                     "state_every_sec", "enable_door_state", "enable_collection_bag", "door_pick_signal")


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """--set имя=значение -> {поле SceneConfig: значение типа поля} (bool: true/false/1/0)."""
    types = {f.name: type(f.default) for f in dataclasses.fields(SceneConfig)}
    out: dict[str, Any] = {}
    for item in items:
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in types:
            raise ValueError(f"--set {item!r}: ожидается поле SceneConfig=значение")
        kind = types[name]
        if kind is bool:
            low = value.strip().lower()
            if low not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"--set {item!r}: bool — true/false")
            out[name] = low in ("true", "1", "yes")
        elif kind in (int, float, str):
            out[name] = kind(value)
        else:
            raise ValueError(f"--set {item!r}: поле типа {kind.__name__} не поддерживается")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Кэш сцен из сырого кэша восприятия, без GPU")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "scene_cache_rebuilt")
    ap.add_argument("--rezone", action="store_true",
                    help="зоны заново текущим кодом по сохранённым детекциям (zero-shot остаётся от старых зон)")
    ap.add_argument("--code-config", action="store_true",
                    help="пороги builder — из текущего кода (SceneConfig по умолчанию), из кэша — только то, "
                         "чем собиралось восприятие: fps, подписи, маска людей, выбор стола")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="ИМЯ=ЗНАЧЕНИЕ",
                    help="поле SceneConfig поверх конфига (повторяемо): --set pose_departure=true")
    args = ap.parse_args(argv)
    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        ap.error(str(exc))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    if not files:
        print(f"в {args.raw} нет сырого кэша *.json", file=sys.stderr)
        return 2
    failed = 0
    for path in files:
        try:
            raw = load_raw(path)
            config = None
            if args.code_config:
                keep = {k: raw["config"][k] for k in PERCEPTION_FIELDS if k in raw.get("config", {})}
                config = dataclasses.replace(SceneConfig(), **keep)
            if overrides:
                if config is None:
                    known = {f.name for f in dataclasses.fields(SceneConfig)}
                    config = SceneConfig(**{k: v for k, v in raw.get("config", {}).items() if k in known})
                config = dataclasses.replace(config, **overrides)
            scene = rebuild_scene(raw, config, rezone=args.rezone)
            if scene.meta.get("door_diff_stale"):
                print(f"{path.stem}: --rezone сменил дверь, door_diff в кэше посчитан для прежней "
                      f"(R15 по нему неверен до пересчёта: scripts/redoor_raw.py или GPU)", file=sys.stderr)
            save_scene(scene, args.out / f"{path.stem}.json")
        except Exception as exc:  # noqa: BLE001 — одно видео не останавливает пересборку
            failed += 1
            print(f"{path.stem}: ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"пересобрано {len(files) - failed} из {len(files)} -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
