#!/usr/bin/env python3
"""Прогон perception + scene по всем видео датасета -> data/scene_cache/<video_id>.json.

Кэш нужен человеку B: на нём подбираются пороги правил без видеокарты (scripts/evaluate.py).
Запускать на машине с GPU после scripts/download_models.py, из корня репозитория:

    PYTHONPATH=. python scripts/build_scene_cache.py
    PYTHONPATH=. python scripts/build_scene_cache.py --only Х-20 --only Х-40 --keep-frames
    PYTHONPATH=. python scripts/build_scene_cache.py --device cpu --fps 2
    PYTHONPATH=. python scripts/build_scene_cache.py --raw-out data/raw_cache   # + сырой кэш для builder

Видео ищутся рекурсивно (*.mov) в --dataset; video_id = имя файла без расширения как есть
(у X-1, X-2 латинская X, у остальных кириллическая Х). Кадры — во временном каталоге
--work/<ascii-имя>/frames, после прогона удаляются (если нет --keep-frames).
Сводка — <out>/_summary.csv: время обработки, найден ли кассир, какие зоны, предупреждения.
Код возврата: 0 — все видео обработаны; 1 — были ошибки; 2 — видео не найдены.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))            # чтобы скрипт работал и без PYTHONPATH=.

from worker.scene import SceneConfig, build_scene, save_scene  # noqa: E402

LOG = logging.getLogger("build_scene_cache")

VIDEO_SUFFIXES = {".mov"}
SUMMARY_NAME = "_summary.csv"
SUMMARY_FIELDS = (
    "video_id", "status", "duration_sec", "elapsed_sec", "states", "cashier_found",
    "zones_found", "entered_t", "exited_t", "warnings",
)


def natural_key(path: Path) -> tuple:
    """Х-2 раньше Х-10."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem))


def find_videos(dataset: Path) -> list[Path]:
    return sorted(
        (p for p in Path(dataset).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=natural_key,
    )


def safe_dirname(video_id: str) -> str:
    """ASCII-имя рабочего каталога (кириллическая Х -> X) + короткий хэш против совпадений."""
    ascii_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id.replace("Х", "X").replace("х", "x")).strip("_") or "video"
    return f"{ascii_part}_{hashlib.md5(video_id.encode('utf-8')).hexdigest()[:6]}"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_summary(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:      # utf-8-sig — чтобы Excel понял кириллицу
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Кэш SceneTimeline по всем видео датасета (для подбора порогов без GPU).")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset", help="папка с видео (поиск рекурсивный)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "scene_cache", help="куда писать <video_id>.json")
    p.add_argument("--work", type=Path, default=ROOT / "data" / "scene_work", help="временные кадры")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models", help="веса из scripts/download_models.py")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--fps", type=float, default=SceneConfig().fps, help="частота анализа кадров")
    p.add_argument("--only", action="append", metavar="VIDEO_ID", help="только это видео (флаг повторяемый)")
    p.add_argument("--keep-frames", action="store_true", help="не удалять кадры (для scripts/viz_scene.py)")
    p.add_argument("--prompts", choices=("v1", "v2"), default="v1",
                   help="подписи zero-shot: v2 — минимальные пары (worker/perception/states.py)")
    p.add_argument("--mask-people", action="store_true", help="закрашивать людей в вырезке стола перед SigLIP")
    p.add_argument("--desk-by-people", action="store_true", help="рабочий стол — тот, где сидели люди, а не самый заметный")
    p.add_argument("--raw-out", type=Path, default=None,
                   help="каталог сырого кэша (data/raw_cache): правки builder потом без GPU, scripts/rebuild_from_raw.py")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    videos = find_videos(args.dataset)
    if args.only:
        wanted = set(args.only)
        videos = [v for v in videos if v.stem in wanted]
        missing = wanted - {v.stem for v in videos}
        if missing:
            LOG.warning("не найдены в %s: %s", args.dataset, ", ".join(sorted(missing)))
    if not videos:
        LOG.error("видео (*.mov) не найдены в %s", args.dataset)
        return 2

    device = resolve_device(args.device)
    config = SceneConfig(fps=args.fps, state_prompts=args.prompts, mask_people_in_desk=args.mask_people,
                         desk_by_people=args.desk_by_people)
    LOG.info("видео: %d, device=%s, fps=%.1f, веса: %s", len(videos), device, args.fps, args.models_dir)

    rows: list[dict] = []
    failed: list[str] = []
    total_started = time.monotonic()
    for i, video in enumerate(videos, 1):
        vid = video.stem
        work = args.work / safe_dirname(vid)
        started = time.monotonic()
        try:
            extra = {"raw_out": args.raw_out / f"{vid}.json"} if args.raw_out else {}
            scene = build_scene(video, work, args.models_dir, device, video_id=vid, config=config, **extra)
            out = save_scene(scene, args.out / f"{vid}.json")
        except Exception as exc:  # noqa: BLE001 — одно видео не должно останавливать прогон
            LOG.error("[%d/%d] %s: ошибка: %s: %s", i, len(videos), vid, type(exc).__name__, exc)
            failed.append(vid)
            rows.append({"video_id": vid, "status": f"error: {type(exc).__name__}: {exc}"})
            continue
        finally:
            if not args.keep_frames:
                shutil.rmtree(work, ignore_errors=True)
        elapsed = time.monotonic() - started
        info = scene.meta.get("scene", {})
        rows.append({
            "video_id": vid,
            "status": "ok",
            "duration_sec": scene.duration_sec,
            "elapsed_sec": round(elapsed, 1),
            "states": len(scene.states),
            "cashier_found": bool(info.get("cashier_track_ids")),
            "zones_found": ",".join(sorted(k for k, v in scene.zones.items() if v is not None)),
            "entered_t": " ".join(str(t) for t in info.get("entered_t", [])),
            "exited_t": " ".join(str(t) for t in info.get("exited_t", [])),
            "warnings": "; ".join(info.get("warnings", [])),
        })
        LOG.info("[%d/%d] %s: %.1f с видео за %.1f с -> %s", i, len(videos), vid, scene.duration_sec, elapsed, out)

    summary = write_summary(args.out / SUMMARY_NAME, rows)
    ok = [r for r in rows if r["status"] == "ok"]
    slowest = max((r["elapsed_sec"] for r in ok), default=0.0)
    LOG.info("готово за %.0f с: успешно %d, ошибок %d, самое долгое видео %.1f с; сводка: %s",
             time.monotonic() - total_started, len(ok), len(failed), slowest, summary)
    if failed:
        LOG.error("с ошибками: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
