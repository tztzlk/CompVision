#!/usr/bin/env python3
"""Пересобрать кэш сцен из сырого кэша текущим builder — без GPU и моделей (секунды на видео).

    PYTHONPATH=. python scripts/rebuild_from_raw.py [--raw data/raw_cache] [--out data/scene_cache_rebuilt]

Сырой кэш пишет scripts/build_scene_cache.py --raw-out data/raw_cache на машине с GPU.
Пороги SceneConfig берутся из сырого кэша (как при сборке) — кроме тех, что новые в коде:
у них значения по умолчанию. Дальше замер: scripts/eval_cache.py --cache <out>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.scene import save_scene  # noqa: E402
from worker.scene.raw_cache import load_raw, rebuild_scene  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Кэш сцен из сырого кэша восприятия, без GPU")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "scene_cache_rebuilt")
    ap.add_argument("--rezone", action="store_true",
                    help="зоны заново текущим кодом по сохранённым детекциям (zero-shot остаётся от старых зон)")
    args = ap.parse_args(argv)
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
            scene = rebuild_scene(load_raw(path), rezone=args.rezone)
            save_scene(scene, args.out / f"{path.stem}.json")
        except Exception as exc:  # noqa: BLE001 — одно видео не останавливает пересборку
            failed += 1
            print(f"{path.stem}: ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"пересобрано {len(files) - failed} из {len(files)} -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
