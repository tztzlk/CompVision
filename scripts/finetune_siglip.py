#!/usr/bin/env python3
"""Частичное дообучение визуальной башни SigLIP под головы состояний зон (safe_open + desk_items).

    PYTHONPATH=. .venv/bin/python scripts/finetune_siglip.py bench --blocks 2 4 --ckpt both
    PYTHONPATH=. .venv/bin/python scripts/finetune_siglip.py cache                       # вырезки один раз (CPU)
    PYTHONPATH=. .venv/bin/python scripts/finetune_siglip.py cv --blocks 4 --folds-json <video->fold.json>
    PYTHONPATH=. .venv/bin/python scripts/finetune_siglip.py final --blocks 4 --epochs 3

Модель — та же google/siglip-base-patch16-224, что в конвейере (models/hf, local_files_only), и та же
предобработка, что у SiglipImageProcessor: RGB, resize 224×224 бикубикой PIL без сохранения пропорций,
x/255, (x − 0,5)/0,5. Вырезка зоны — как worker.perception.states.crop_zone с полем ZONE_CROP_MARGIN.
Эмбеддинг — выход головы attention pooling (= get_image_features), L2-нормированный; поверх — линейная
многозадачная голова: логит задачи = emb @ w + b, как у логрегрессий models/heads/*.npz (тёплый старт —
fit_logreg из scripts/train_heads.py на замороженных эмбеддингах data/emb того же обучающего набора).

Обучаются последние N блоков энкодера, post_layernorm и голова attention pooling (она стоит после
дообучаемых блоков и определяет, какие патчи попадут в эмбеддинг, — без неё сдвиг признаков блоков
пришлось бы ловить одной линейной головой; по памяти она почти бесплатна: активаций одного запроса
мало, 7 млн параметров). Замороженная часть — fp16 и под no_grad, её активации не хранятся.

Лосс — BCE по задачам с маской неразмеченных (вырезка сейфа учит только safe_open, вырезка стола —
только классы стола; «?» и occluded — маска 0) и весами классов как в fit_logreg: n/(2·n_pos), n/(2·n_neg)
по каждой задаче на обучающем наборе.

Проверка — только фолды по видео: JSON video->fold базы P2 (--folds-json), иначе свои фолды по правилам
train_heads.assign_folds (Х-4≡Х-47, Х-5≡Х-48 — одна группа). Офлайн-варианты flip/persp/crop/light
(scripts/augment_videos.py) — в фолде оригинала: в обучение только варианты обучающих видео
(--train-variants orig — варианты только для оценки). Ранняя остановка — по внутреннему валидационному
фолду (следующий фолд из обучающих), тестовый фолд в выборе эпохи не участвует.

Кэш вырезок (cache) -> data/ft_crops/<zone>/<variant>/<video_id>.npz: t, eval (JPEG точной вырезки
конвейера после resize 224×224 процессора), ctx (JPEG рамки с запасом --ctx-margin под онлайн-сдвиг,
--ctx-size²), rel (рамка вырезки конвейера внутри ctx, доли). Кадры заново не извлекаются: data/frames_r2.

Выход cv -> data/ml_results/r4/ft/<run>/: <task>_oof.json ({вариант: {видео: [[t, p, y], ...]}} — как
train_heads), folds.json, report.json. final -> models/ft/siglip_states/: state.pt (только дообученные
блоки + post_layernorm + pooling + линейная голова, fp16) + meta.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cache import known_aliases, normalize_video_id  # noqa: E402
from worker.perception.geometry import expand  # noqa: E402
from worker.perception.states import ZONE_CROP_MARGIN  # noqa: E402

SIZE = 224
VARIANTS = ("orig", "flip", "persp", "crop", "light")
# cache_<v> — кадры и зоны аугментированных кэшей (data/scene_work_aug_<v>, data/raw_cache_aug_<v>), как их
# видит конвейер; только для оценки (база P2 — data/emb_aug/<v>/<zone>/orig).
CACHE_VARIANTS = tuple(f"cache_{v}" for v in VARIANTS[1:])
ALL_VARIANTS = VARIANTS + CACHE_VARIANTS
DESK_CLASSES = ("cash", "bag", "coat", "cup")
# Задачи многозадачной головы: (имя, зона).
TASKS: tuple[tuple[str, str], ...] = (("safe_open", "safe"),) + tuple((f"desk_{c}", "desk") for c in DESK_CLASSES)
TASK_NAMES = tuple(t for t, _ in TASKS)
LOCK_HINT = "flock <каталог>/gpu.lock"


def models_dir() -> Path:
    """models/ основной копии: в worktree это каталог с симлинком models/models."""
    for cand in (ROOT / "models", ROOT / "models" / "models"):
        if (cand / "hf" / "siglip-base-patch16-224" / "config.json").is_file():
            return cand
    return ROOT / "models"


def siglip_path() -> Path:
    return models_dir() / "hf" / "siglip-base-patch16-224"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


# ---------------------------------------------------------------- предобработка (как SiglipImageProcessor)

def processor_resize(rgb: np.ndarray) -> np.ndarray:
    """resize SiglipImageProcessor: PIL BICUBIC до 224×224 без сохранения пропорций, uint8 RGB."""
    from PIL import Image

    return np.array(Image.fromarray(rgb).resize((SIZE, SIZE), Image.BICUBIC))


def to_pixels(batch_rgb: np.ndarray) -> Any:
    """N×224×224×3 uint8 RGB -> N×3×224×224 float: x/255, (x − 0,5)/0,5."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(batch_rgb)).permute(0, 3, 1, 2).float()
    return (x / 255.0 - 0.5) / 0.5


def _px_box(box: tuple[float, ...], w: int, h: int) -> tuple[int, int, int, int]:
    """Пиксели рамки — ровно как crop_zone."""
    return int(box[0] * w), int(box[1] * h), int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))


def _jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _unjpeg(data: bytes) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- кэш вырезок

def crops_for_video(raw: dict[str, Any], frames_dir: Path, zone: str, variant: str, every: float,
                    ctx_margin: float, ctx_size: int, quality: int, work_layout: bool = False,
                    lossless: bool = False, times_ms: Optional[set[int]] = None) -> Optional[dict[str, Any]]:
    """Вырезки одного видео/варианта: точная вырезка конвейера (как iter_crops) + рамка с запасом.
    lossless — только точная вырезка после resize процессора, uint8 без JPEG (для валидации/теста)."""
    from augment_videos import make_transform, transform_bbox, variant_params
    from worker.perception.video import imread_unicode

    bbox = raw["zones"].get(zone)
    if bbox is None:
        return None
    w, h = raw["frame_size"]
    fn = None
    if variant in VARIANTS[1:]:
        params = variant_params(variant)
        fn = make_transform(variant, params, w, h)
        bbox = transform_bbox(variant, params, bbox, w, h)
    pbox = expand(tuple(bbox), ZONE_CROP_MARGIN.get(zone, 0.1))
    cbox = expand(pbox, ctx_margin)
    ts, evals, ctxs, rels = [], [], [], []
    next_t = 0.0
    for idx, t, rel_path in raw["frames"]:
        if times_ms is not None:
            if int(round(float(t) * 1000)) not in times_ms:
                continue
        elif t + 1e-6 < next_t:
            continue
        image = imread_unicode(frames_dir / rel_path if work_layout else frames_dir / f"{int(idx):06d}.jpg")
        if image is None:
            continue
        next_t = t + every
        if fn is not None:
            image = fn(image)
        ih, iw = image.shape[:2]
        x1, y1, x2, y2 = _px_box(pbox, iw, ih)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        rgb = image[:, :, ::-1]
        exact = processor_resize(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
        ts.append(float(t))
        if lossless:
            evals.append(exact)
            continue
        cx1, cy1, cx2, cy2 = _px_box(cbox, iw, ih)
        evals.append(_jpeg(exact, quality))
        ctx = cv2.resize(np.ascontiguousarray(rgb[cy1:cy2, cx1:cx2]), (ctx_size, ctx_size), interpolation=cv2.INTER_AREA
                         if (cx2 - cx1) > ctx_size else cv2.INTER_CUBIC)
        ctxs.append(_jpeg(ctx, quality))
        cw, ch = cx2 - cx1, cy2 - cy1
        rels.append(((x1 - cx1) / cw, (y1 - cy1) / ch, (x2 - cx1) / cw, (y2 - cy1) / ch))
    if not ts:
        return None
    if lossless:
        return {"t": np.array(ts, dtype=np.float32), "exact": np.stack(evals)}
    return {"t": np.array(ts, dtype=np.float32), "eval": evals, "ctx": ctxs, "rel": np.array(rels, dtype=np.float32)}


def _pack(blobs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    off = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    return np.frombuffer(b"".join(blobs), dtype=np.uint8), off


def _unpack(data: np.ndarray, off: np.ndarray, i: int) -> bytes:
    return data[off[i]:off[i + 1]].tobytes()


def _frames_dir(root: Path, variant: str, video_id: str) -> tuple[Path, bool]:
    """Каталог кадров: frames_r2/<video_id> или рабочий каталог аугментированного кэша (cache_<v>)."""
    if variant.startswith("cache_"):
        from build_scene_cache import safe_dirname

        return ROOT / "data" / f"scene_work_aug_{variant[6:]}" / safe_dirname(video_id), True
    return root / video_id, False


def _cache_job(job: tuple) -> tuple[str, int, int]:
    raw, frames_root, zone, variant, every, ctx_margin, ctx_size, quality, target = job
    fdir, work = _frames_dir(frames_root, variant, raw["video_id"])
    out = crops_for_video(raw, fdir, zone, variant, every, ctx_margin, ctx_size, quality, work)
    if out is None:
        return str(target), 0, 0
    ev, ev_off = _pack(out["eval"])
    cx, cx_off = _pack(out["ctx"])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, t=out["t"], rel=out["rel"], eval=ev, eval_off=ev_off, ctx=cx, ctx_off=cx_off,
             ctx_size=np.int32(ctx_size), ctx_margin=np.float32(ctx_margin))
    return str(target), len(out["t"]), int(ev.nbytes + cx.nbytes)


def _exact_job(job: tuple) -> tuple[tuple[str, str, str], Optional[dict[str, Any]]]:
    key, raw, frames_root, every = job
    zone, variant, _vid = key
    fdir, work = _frames_dir(frames_root, variant, raw["video_id"])
    return key, crops_for_video(raw, fdir, zone, variant, every, 0.0, 0, 0, work, lossless=True)


def exact_crops(items: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    """Вырезки для валидации/теста без потерь: заново из исходных кадров (frames_r2 + преобразование варианта,
    кадры scene_work_aug_<v> + зона raw_cache_aug_<v> для cache_<v>) — ровно вход конвейера. N×224×224×3 uint8."""
    from multiprocessing import Pool

    from embed_zones import load_raw

    keys = sorted({(it["zone"], it["variant"], it["raw_id"]) for it in items})
    raws: dict[tuple[str, str], dict[str, Any]] = {}
    jobs = []
    for zone, variant, raw_id in keys:
        src = args.raw if not variant.startswith("cache_") else ROOT / "data" / f"raw_cache_aug_{variant[6:]}"
        if (str(src), raw_id) not in raws:
            raws[(str(src), raw_id)] = json.loads((src / f"{raw_id}.json").read_text(encoding="utf-8"))
        jobs.append(((zone, variant, raw_id), raws[(str(src), raw_id)], args.frames, args.every))
    by_key: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    with Pool(min(4, max(1, len(jobs)))) as pool:
        for key, out in pool.imap_unordered(_exact_job, jobs):
            by_key[key] = {} if out is None else {int(round(float(t) * 1000)): im for t, im in zip(out["t"], out["exact"])}
    res = np.zeros((len(items), SIZE, SIZE, 3), dtype=np.uint8)
    for n, it in enumerate(items):
        res[n] = by_key[(it["zone"], it["variant"], it["raw_id"])][int(round(it["t"] * 1000))]
    return res


SAMPLE_KEYS = {"safe": "safe_open", "desk": "cash_on_desk"}   # ряды state_samples сырого кэша по зоне


def _sample_job(job: tuple) -> tuple[tuple[str, str, str], Optional[dict[str, Any]]]:
    key, raw, frames_root, times = job
    zone, variant, _vid = key
    fdir, work = _frames_dir(frames_root, variant, raw["video_id"])
    return key, crops_for_video(raw, fdir, zone, variant, 1.0, 0.0, 0, 0, work, lossless=True, times_ms=times)


def predict_samples(model: Any, videos: set[str], fold: int, args: argparse.Namespace,
                    label_sets: dict[str, dict[str, list]]) -> list[dict[str, Any]]:
    """Предсказания модели фолда на кадрах, где конвейер сэмплирует состояния: времена state_samples
    (safe_open — зона safe, cash_on_desk — зона desk; скан OWLv2 item_scan — их подмножество, флаг scan)
    из raw_cache_r3 (orig) и raw_cache_aug_<v> (cache_<v>). Метки — если есть (иначе nan)."""
    from multiprocessing import Pool

    import torch

    rows: list[dict[str, Any]] = []
    jobs, meta = [], {}
    for variant in ("orig",) + CACHE_VARIANTS:
        src = args.raw if variant == "orig" else ROOT / "data" / f"raw_cache_aug_{variant[6:]}"
        for path in sorted(src.glob("*.json")):
            if path.name.startswith("_") or normalize_video_id(path.stem) not in videos:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            scan = {int(round(float(e[0]) * 1000)) for e in raw.get("item_scan") or []}
            for zone, sk in SAMPLE_KEYS.items():
                times = {int(round(float(t) * 1000)) for t, _ in raw["state_samples"].get(sk, [])}
                if not times or raw["zones"].get(zone) is None:
                    continue
                key = (zone, variant, raw["video_id"])
                meta[key] = (str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), scan)
                jobs.append((key, raw, args.frames, times))
    if not jobs:
        return rows
    crops: list[np.ndarray] = []
    with Pool(min(4, len(jobs))) as pool:
        for key, out in pool.imap_unordered(_sample_job, jobs):
            if out is None:
                continue
            zone, variant, raw_id = key
            vid = normalize_video_id(raw_id)
            src, scan = meta[key]
            for t, im in zip(out["t"], out["exact"]):
                y = np.full(len(TASKS), np.nan, dtype=np.float32)
                segs = label_sets.get(zone, {}).get(vid)
                if segs:
                    tt = float(round(float(t))) if (args.desk_label_time if zone == "desk" else args.safe_label_time) == "round" else float(t)
                    lab = labels_at(segs, tt)
                    for k, (task, z) in enumerate(TASKS):
                        if z == zone and task in args.tasks:
                            y[k] = lab.get(task, np.nan)
                rows.append({"zone": zone, "variant": variant, "video": vid, "t": float(t), "src": src,
                             "scan": int(round(float(t) * 1000)) in scan, "fold": fold, "y": y})
                crops.append(im)
    model.eval()
    ps = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(crops), args.eval_batch):
            x = to_pixels(np.stack(crops[s:s + args.eval_batch])).cuda(non_blocking=True)
            ps.append(torch.sigmoid(model(x).float()).cpu().numpy())
    for r, pr in zip(rows, np.concatenate(ps)):
        r["p"] = pr
    return rows


def save_samples(rows: list[dict[str, Any]], out: Path, tasks: list[str]) -> None:
    """<task>_<variant>.npz как ft_oof_mt + src (сырой кэш) и scan (кадр скана OWLv2)."""
    for k, (task, zone) in enumerate(TASKS):
        if task not in tasks:
            continue
        for v in ("orig",) + CACHE_VARIANTS:
            rr = sorted((r for r in rows if r["zone"] == zone and r["variant"] == v), key=lambda r: (r["video"], r["t"]))
            if not rr:
                continue
            np.savez(out / f"{task}_{v}.npz", video=np.array([r["video"] for r in rr]),
                     t=np.array([r["t"] for r in rr], dtype=np.float32),
                     t_ms=np.array([int(round(r["t"] * 1000)) for r in rr], dtype=np.int64),
                     y=np.array([-1 if np.isnan(r["y"][k]) else int(r["y"][k]) for r in rr], dtype=np.int8),
                     p=np.array([r["p"][k] for r in rr], dtype=np.float32),
                     fold=np.array([r["fold"] for r in rr], dtype=np.int8),
                     src=np.array([r["src"] for r in rr]), scan=np.array([r["scan"] for r in rr], dtype=bool))


def cmd_cache(args: argparse.Namespace) -> int:
    from multiprocessing import Pool

    from embed_zones import load_raw

    raws_by: dict[str, list] = {}
    jobs = []
    for zone in args.zones:
        for variant in args.variants:
            src = args.raw if not variant.startswith("cache_") else ROOT / "data" / f"raw_cache_aug_{variant[6:]}"
            if src not in raws_by:
                raws_by[src] = load_raw(src, args.only)
            for raw in raws_by[src]:
                target = args.cache / zone / variant / f"{raw['video_id']}.npz"
                if target.exists() and not args.force:
                    continue
                jobs.append((raw, args.frames, zone, variant, args.every, args.ctx_margin, args.ctx_size,
                             args.quality, target))
    if not jobs:
        print("кэш уже собран")
        check_times(args.cache, args.emb)
        return 0
    # Оценка размера до записи: одна вырезка ~ по пробе первого видео.
    fdir, work = _frames_dir(args.frames, jobs[0][3], jobs[0][0]["video_id"])
    probe = crops_for_video(jobs[0][0], fdir, jobs[0][2], jobs[0][3], args.every,
                            args.ctx_margin, args.ctx_size, args.quality, work)
    per = (sum(map(len, probe["eval"])) + sum(map(len, probe["ctx"]))) / max(len(probe["t"]), 1) if probe else 60e3
    n_est = sum(int(j[0]["duration_sec"] // args.every) + 1 for j in jobs)
    est_mb = n_est * per / 1e6
    print(f"заданий {len(jobs)}, вырезок ≈{n_est}, ≈{per / 1e3:.0f} КБ на вырезку -> ≈{est_mb:.0f} МБ", flush=True)
    if est_mb > args.max_mb:
        print(f"больше --max-mb {args.max_mb}: не пишу", file=sys.stderr)
        return 2
    t0, n, nbytes = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for target, k, b in pool.imap_unordered(_cache_job, jobs):
            n, nbytes = n + k, nbytes + b
    print(f"{n} вырезок, {nbytes / 1e6:.0f} МБ за {time.time() - t0:.0f} с -> {args.cache}", flush=True)
    check_times(args.cache, args.emb)
    return 0


def check_times(cache: Path, emb_root: Path) -> None:
    """Время вырезок кэша — подмножество времени data/emb (база может быть с шагом мельче) — по ключам в мс."""
    bad = 0
    for path in sorted(cache.glob("*/*/*.npz")):
        zone, variant = path.parent.parent.name, path.parent.name
        if variant == "orig":
            ref = emb_root / zone / "orig" / path.name
        elif variant.startswith("cache_"):
            ref = ROOT / "data" / "emb_aug" / variant[6:] / zone / "orig" / path.name
        else:
            continue
        if not ref.exists():
            continue
        a = {int(round(float(t) * 1000)) for t in np.load(path)["t"]}
        b = {int(round(float(t) * 1000)) for t in np.load(ref)["t"]}
        if a - b:
            bad += 1
            print(f"  нет в базе: {zone}/{variant}/{path.stem}: {len(a - b)} ключей")
    print("время вырезок есть в data/emb" if not bad else f"расхождений времени: {bad} видео")


# ---------------------------------------------------------------- метки и данные

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def safe_labels(path: Path) -> dict[str, list[tuple[float, float, dict[str, float]]]]:
    """safe_open.csv -> видео -> [(t0, t1, {safe_open: 1/0/nan})]; occluded — nan (маска)."""
    out: dict[str, list] = defaultdict(list)
    for r in read_csv_rows(path):
        s = (r.get("state") or "").strip()
        val = 1.0 if s == "open" else 0.0 if s == "closed" else float("nan")
        out[normalize_video_id(r["video_id"])].append((float(r["t_from"]), float(r["t_to"]), {"safe_open": val}))
    return out


def _desk_value(cell: str) -> float:
    c = (cell or "").strip().lower()
    if c in ("1", "yes", "y", "+", "true", "да"):
        return 1.0
    if c in ("0", "no", "n", "-", "false", "нет", ""):
        return 0.0
    return float("nan")      # «?» и прочее — не учим


def desk_labels(path: Path) -> dict[str, list[tuple[float, float, dict[str, float]]]]:
    """desk_items.csv (P1). Поддержаны два вида:
    - столбцы cash,bag,coat,cup со значениями 1/0/? (пусто = 0);
    - один столбец items/state/labels со списком классов через ; + , или пробел; «?» — весь интервал не учим,
      «cash?» — не учим только cash; none/пусто/- — ни одного предмета.
    Строки видео применяются по порядку, следующая перекрывает предыдущую (как safe_open.csv)."""
    out: dict[str, list] = defaultdict(list)
    for r in read_csv_rows(path):
        keys = {k.strip().lower(): v for k, v in r.items() if k}
        if any(c in keys for c in DESK_CLASSES):
            vals = {f"desk_{c}": _desk_value(keys.get(c, "")) for c in DESK_CLASSES}
        else:
            cell = next((keys[k] for k in ("items", "state", "labels", "classes") if k in keys), "") or ""
            toks = [t for t in cell.replace(";", " ").replace("+", " ").replace(",", " ").replace("|", " ").split() if t]
            if "?" in toks:
                vals = {f"desk_{c}": float("nan") for c in DESK_CLASSES}
            else:
                vals = {f"desk_{c}": 0.0 for c in DESK_CLASSES}
                for t in toks:
                    t = t.lower()
                    name = t.rstrip("?")
                    if name in DESK_CLASSES:
                        vals[f"desk_{name}"] = float("nan") if t.endswith("?") else 1.0
        out[normalize_video_id(r["video_id"])].append((float(r["t_from"]), float(r["t_to"]), vals))
    return out


def labels_at(segs: list[tuple[float, float, dict[str, float]]], t: float) -> dict[str, float]:
    state: dict[str, float] = {}
    for t0, t1, vals in segs:
        if t0 - 1e-6 <= t <= t1 + 1e-6:
            state.update(vals)
    return state


def group_of(vid: str, aliases: dict[str, tuple[str, ...]]) -> str:
    from train_heads import group_of as _g

    return _g(vid, aliases)


def load_items(cache: Path, label_files: dict[str, Path],
               label_time: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
    """Все вырезки кэша с размеченными задачами: y (len TASKS, nan = не учим).

    label_time[zone]: exact — метка по точному t (как train_heads.load_data у safe_open: база P2 safe_open
    совпадает 1434/1434), round — по round(t) (как P2 desk_items: листы подписаны целой секундой)."""
    label_time = label_time or {"safe": "exact", "desk": "round"}
    labels: dict[str, dict[str, list]] = {}
    if label_files.get("safe") and label_files["safe"].exists():
        labels["safe"] = safe_labels(label_files["safe"])
    if label_files.get("desk") and label_files["desk"].exists():
        labels["desk"] = desk_labels(label_files["desk"])
    aliases = {normalize_video_id(k): v for k, v in known_aliases().items()}
    items = []
    for zone, zlab in labels.items():
        zone_tasks = [i for i, (_, z) in enumerate(TASKS) if z == zone]
        for variant in ALL_VARIANTS:
            for path in sorted((cache / zone / variant).glob("*.npz")):
                vid = normalize_video_id(path.stem)
                segs = zlab.get(vid)
                if not segs:
                    continue
                data = np.load(path)
                for i, t in enumerate(data["t"]):
                    tt = float(round(float(t))) if label_time.get(zone) == "round" else float(t)
                    lab = labels_at(segs, tt)
                    y = np.full(len(TASKS), np.nan, dtype=np.float32)
                    for k in zone_tasks:
                        y[k] = lab.get(TASKS[k][0], np.nan)
                    items.append({"zone": zone, "variant": variant, "video": vid, "group": group_of(vid, aliases),
                                  "raw_id": path.stem, "t": float(t), "path": str(path), "i": i, "y": y})
    return items


class CropStore:
    """Ленивая загрузка JPEG из npz кэша: в процессе DataLoader держится не больше max_open видео (~1 МБ каждое),
    а не весь кэш."""

    def __init__(self, max_open: int = 8) -> None:
        from collections import OrderedDict

        self._open: "OrderedDict[str, Any]" = OrderedDict()
        self.max_open = max_open

    def get(self, path: str) -> Any:
        d = self._open.pop(path, None)
        if d is None:
            z = np.load(path)
            d = {k: z[k] for k in ("eval", "eval_off", "ctx", "ctx_off", "rel")}
        self._open[path] = d
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return d


# ---------------------------------------------------------------- онлайн-аугментации

def augment(ctx: np.ndarray, rel: np.ndarray, rng: random.Random, shift: float = 0.12, persp: float = 0.06) -> np.ndarray:
    """Рамка вырезки конвейера внутри ctx -> сдвиг/масштаб ±shift, перспектива ±persp (доли рамки), hflip,
    яркость/контраст/гамма, blur, JPEG. Выход — 224×224 uint8 RGB."""
    S = ctx.shape[0]
    x1, y1, x2, y2 = (float(v) * S for v in rel)
    bw, bh = x2 - x1, y2 - y1
    sc = 1.0 + rng.uniform(-shift, shift)
    cx = (x1 + x2) / 2 + rng.uniform(-shift, shift) * bw
    cy = (y1 + y2) / 2 + rng.uniform(-shift, shift) * bh
    hw, hh = bw * sc / 2, bh * sc * (1.0 + rng.uniform(-0.05, 0.05)) / 2
    src = np.float32([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
    if rng.random() < 0.6:
        src += np.float32([[rng.uniform(-persp, persp) * bw, rng.uniform(-persp, persp) * bh] for _ in range(4)])
    dst = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    m = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(ctx, m, (SIZE, SIZE), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.8:
        x = img.astype(np.float32) / 255.0
        g = math.exp(rng.uniform(-0.35, 0.35))
        x = np.power(np.clip(x, 0, 1), g)
        c, b = rng.uniform(0.75, 1.25), rng.uniform(-0.12, 0.12)
        x = (x - 0.5) * c + 0.5 + b
        x = x * np.float32([rng.uniform(0.94, 1.06) for _ in range(3)])
        img = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)])
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return np.ascontiguousarray(img)


class CropDataset:
    """train — аугментации из ctx, с вероятностью p_clean — точная вырезка кэша без аугментаций (вход конвейера,
    JPEG q95); оценка — только exact (без потерь, exact_crops)."""

    def __init__(self, items: list[dict[str, Any]], train: bool, seed: int = 0, p_clean: float = 0.3,
                 exact: Optional[np.ndarray] = None) -> None:
        self.items, self.train, self.seed, self.p_clean, self.exact = items, train, seed, p_clean, exact
        self.store = CropStore()
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> tuple[Any, Any]:
        import torch

        it = self.items[k]
        if self.exact is not None:
            img = self.exact[k]
        else:
            d = self.store.get(it["path"])
            i = it["i"]
            rng = random.Random(hash((self.seed, self.epoch, k)) & 0xFFFFFFFF)
            if not self.train or rng.random() < self.p_clean:
                img = _unjpeg(_unpack(d["eval"], d["eval_off"], i))
            else:
                img = augment(_unjpeg(_unpack(d["ctx"], d["ctx_off"], i)), d["rel"][i], rng)
        x = to_pixels(img[None])[0]
        return x, torch.from_numpy(it["y"])


# ---------------------------------------------------------------- модель

def build_model(n_blocks: int, grad_ckpt: bool, device: str, train_pool: bool = True) -> Any:
    import torch
    from torch import nn
    from transformers import SiglipVisionModel

    vm = SiglipVisionModel.from_pretrained(siglip_path(), local_files_only=True, attn_implementation="sdpa").vision_model

    class FT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vm = vm
            layers = vm.encoder.layers
            self.k = len(layers) - n_blocks
            self.grad_ckpt = grad_ckpt
            for p in vm.parameters():
                p.requires_grad_(False)
            trainable = list(layers[self.k:]) + [vm.post_layernorm] + ([vm.head] if train_pool else [])
            for mod in trainable:
                for p in mod.parameters():
                    p.requires_grad_(True)
            # Замороженная часть — fp16 (под autocast всё равно считается в fp16), памяти вдвое меньше.
            vm.embeddings.half()
            for layer in layers[:self.k]:
                layer.half()
            if not train_pool:
                vm.head.half()
            self.cls = nn.Linear(vm.config.hidden_size, len(TASKS))
            nn.init.zeros_(self.cls.weight)
            nn.init.zeros_(self.cls.bias)

        def frozen_part(self, px: Any) -> Any:
            with torch.no_grad():
                h = self.vm.embeddings(px.to(self.vm.embeddings.patch_embedding.weight.dtype))
                for layer in self.vm.encoder.layers[:self.k]:
                    h = layer(h, None)[0]
            # Обучаемые блоки — fp32: без autocast (CPU) тип должен совпадать; под autocast всё равно fp16.
            return h.float()

        def embed(self, px: Any) -> Any:
            from torch.utils.checkpoint import checkpoint

            h = self.frozen_part(px)
            for layer in self.vm.encoder.layers[self.k:]:
                if self.grad_ckpt and self.training:
                    h = checkpoint(lambda x, lay=layer: lay(x, None)[0], h, use_reentrant=False)
                else:
                    h = layer(h, None)[0]
            h = self.vm.post_layernorm(h)
            e = self.vm.head(h)
            return torch.nn.functional.normalize(e.float(), dim=-1)

        def forward(self, px: Any) -> Any:
            return self.cls(self.embed(px))

        def trainable_state(self) -> dict[str, Any]:
            names = {n for n, p in self.named_parameters() if p.requires_grad}
            return {n: t.detach().half().cpu() for n, t in self.state_dict().items()
                    if n in names or n.startswith("cls.")}

    model = FT().to(device)
    return model


def param_groups(model: Any, lr_blocks: float, lr_head: float, wd: float) -> list[dict[str, Any]]:
    blocks = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls.")]
    return [{"params": [p for _, p in blocks if p.ndim >= 2], "lr": lr_blocks, "weight_decay": wd},
            {"params": [p for _, p in blocks if p.ndim < 2], "lr": lr_blocks, "weight_decay": 0.0},   # LayerNorm, bias
            {"params": list(model.cls.parameters()), "lr": lr_head, "weight_decay": 0.0}]


def limit_memory(mem_gb: float) -> None:
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, mem_gb * 1024 ** 3 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"лимит аллокатора torch: {mem_gb:.2f} ГБ ({frac:.3f} от {total / 1024 ** 3:.2f} ГБ)", flush=True)


def masked_bce(logits: Any, y: Any, pos_w: Any, neg_w: Any) -> Any:
    """BCE по задачам с маской nan и весами классов fit_logreg."""
    import torch

    mask = ~torch.isnan(y)
    yy = torch.nan_to_num(y, nan=0.0)
    w = torch.where(yy > 0.5, pos_w, neg_w) * mask
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), yy, weight=w, reduction="sum")
    return loss / mask.sum().clamp(min=1)


def class_weights(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Как fit_logreg: n/(2·n_pos) и n/(2·n_neg) по размеченным каждой задачи."""
    pos_w, neg_w = np.ones(y.shape[1], np.float32), np.ones(y.shape[1], np.float32)
    for k in range(y.shape[1]):
        col = y[:, k][~np.isnan(y[:, k])]
        if len(col):
            n_pos, n_neg = float(col.sum()), float(len(col) - col.sum())
            pos_w[k], neg_w[k] = len(col) / (2 * max(n_pos, 1)), len(col) / (2 * max(n_neg, 1))
    return pos_w, neg_w


# ---------------------------------------------------------------- bench

def cmd_bench(args: argparse.Namespace) -> int:
    import torch

    free0, total = torch.cuda.mem_get_info()   # до создания контекста CUDA не узнать — см. process_mb
    limit_memory(args.mem_gb)
    print(f"GPU свободно до старта: {free0 / 1024 ** 3:.2f} из {total / 1024 ** 3:.2f} ГБ", flush=True)
    rows = []
    ckpts = {"off": [False], "on": [True], "both": [False, True]}[args.ckpt]
    for n in args.blocks:
        for ck in ckpts:
            for bs in args.batch:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    model = build_model(n, ck, "cuda")
                    model.train()
                    opt = torch.optim.AdamW(param_groups(model, 1e-5, 1e-3, 0.01))
                    scaler = torch.amp.GradScaler("cuda")
                    px = torch.randn(bs, 3, SIZE, SIZE, device="cuda").clamp(-1, 1)
                    y = torch.randint(0, 2, (bs, len(TASKS)), device="cuda").float()
                    y[:, 1:] = float("nan")
                    pw = torch.ones(len(TASKS), device="cuda")
                    times = []
                    for step in range(args.steps + 3):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.float16):
                            loss = masked_bce(model(px), y, pw, pw)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                        torch.cuda.synchronize()
                        if step >= 3:
                            times.append(time.perf_counter() - t0)
                    model.eval()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(3):
                            model(torch.randn(64, 3, SIZE, SIZE, device="cuda"))
                        torch.cuda.synchronize()
                        inf = (time.perf_counter() - t0) / (3 * 64)
                    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
                    # Всё, что процесс занял на карте (контекст CUDA + кэш аллокатора) — для соседства с worker.
                    proc_mb = (free0 - torch.cuda.mem_get_info()[0]) / 1024 ** 2
                    reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
                    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    row = {"blocks": n, "ckpt": ck, "batch": bs, "step_s": float(np.median(times)),
                           "peak_alloc_mb": round(peak), "peak_reserved_mb": round(reserved),
                           "trainable_m": round(n_tr / 1e6, 1), "process_mb": round(proc_mb), "infer_ms_per_crop": round(inf * 1000, 2)}
                except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    row = {"blocks": n, "ckpt": ck, "batch": bs, "oom": str(e).splitlines()[0][:120]}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                rows.append(row)
                model = opt = scaler = px = None  # noqa: F841
                import gc

                gc.collect()
                torch.cuda.empty_cache()
    # Подготовка батча на CPU (аугментации) — отдельно от GPU.
    rng = random.Random(0)
    ctx = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    rel = np.float32([0.115, 0.115, 0.885, 0.885])
    t0 = time.perf_counter()
    for _ in range(64):
        to_pixels(augment(ctx, rel, rng)[None])
    aug_ms = (time.perf_counter() - t0) / 64 * 1000
    print(json.dumps({"cpu_aug_ms_per_crop_1_thread": round(aug_ms, 2)}), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "cpu_aug_ms": aug_ms, "mem_gb": args.mem_gb,
                                        "free_gb_before": free0 / 1024 ** 3}, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- обучение

def make_loader(items: list[dict[str, Any]], train: bool, args: argparse.Namespace, seed: int = 0) -> Any:
    import torch

    ds = CropDataset(items, train, seed, args.p_clean)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=train,
                                       num_workers=args.workers, drop_last=train and len(items) > args.batch,
                                       persistent_workers=False, pin_memory=False)


def predict(model: Any, items: list[dict[str, Any]], args: argparse.Namespace,
            exact: Optional[np.ndarray] = None) -> np.ndarray:
    """Вероятности задач; вход — exact (без потерь), иначе вырезается из кадров здесь же."""
    import torch

    if exact is None:
        exact = exact_crops(items, args)
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(items), args.eval_batch):
            x = to_pixels(exact[s:s + args.eval_batch]).cuda(non_blocking=True)
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TASKS)), np.float32)


def eval_loss(model: Any, items: list[dict[str, Any]], args: argparse.Namespace, pw: Any, nw: Any,
              exact: Optional[np.ndarray] = None) -> float:
    """Взвешенный BCE (те же веса) на валидационных вырезках оригинала."""
    import torch

    p = predict(model, items, args, exact)
    y = np.stack([it["y"] for it in items])
    with torch.no_grad():
        logits = torch.logit(torch.from_numpy(p).clamp(1e-6, 1 - 1e-6))
        return float(masked_bce(logits, torch.from_numpy(y), pw.cpu(), nw.cpu()))


def warm_start(model: Any, train_items: list[dict[str, Any]], emb_root: Path, l2: float) -> dict[str, Any]:
    """Линейная голова: fit_logreg (train_heads) на замороженных эмбеддингах data/emb тех же вырезок."""
    import torch
    from train_heads import fit_logreg

    cache: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    info = {}
    for k, (task, zone) in enumerate(TASKS):
        xs, ys = [], []
        for it in train_items:
            if it["zone"] != zone or np.isnan(it["y"][k]):
                continue
            key = (zone, it["variant"], it["video"])
            if key not in cache:
                path = next((p for p in (emb_root / zone / it["variant"]).glob("*.npz")
                             if normalize_video_id(p.stem) == it["video"]), None)
                cache[key] = ({int(round(float(t) * 1000)): e for t, e in zip(*(np.load(path)[n] for n in ("t", "emb")))}
                              if path else {})
            e = cache[key].get(int(round(it["t"] * 1000)))
            if e is not None:
                xs.append(e)
                ys.append(it["y"][k])
        ys_a = np.array(ys)
        if len(ys) < 10 or ys_a.sum() == 0 or ys_a.sum() == len(ys_a):
            # Нет замороженных эмбеддингов — нулевые веса и сдвиг на априорную долю.
            prior = float(np.nanmean([it["y"][k] for it in train_items if it["zone"] == zone] or [0.5]))
            prior = min(max(prior if not math.isnan(prior) else 0.5, 1e-3), 1 - 1e-3)
            with torch.no_grad():
                model.cls.bias[k] = 0.0  # веса классов уравновешивают классы — сдвиг 0
            info[task] = {"init": "zero", "prior": prior, "n_emb": len(ys)}
            continue
        w, b = fit_logreg(np.stack(xs), ys_a, l2)
        with torch.no_grad():
            model.cls.weight[k] = torch.from_numpy(w).to(model.cls.weight.device)
            model.cls.bias[k] = float(b)
        info[task] = {"init": "fit_logreg", "l2": l2, "n_emb": len(ys)}
    return info


def train_one(train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], args: argparse.Namespace,
              seed: int) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(args.blocks, args.grad_ckpt, "cuda", not args.freeze_pool)
    log: dict[str, Any] = {"warm_start": warm_start(model, train_items, args.emb, args.init_l2), "epochs": []}
    y = np.stack([it["y"] for it in train_items])
    pw_np, nw_np = class_weights(y)
    pw, nw = torch.from_numpy(pw_np).cuda(), torch.from_numpy(nw_np).cuda()
    log["class_weights"] = {t: [round(float(a), 3), round(float(b), 3)] for t, a, b in zip(TASK_NAMES, pw_np, nw_np)}
    opt = torch.optim.AdamW(param_groups(model, args.lr_blocks, args.lr_head, args.wd))
    scaler = torch.amp.GradScaler("cuda")
    loader = make_loader(train_items, True, args, seed)
    steps_per_epoch = max(1, len(loader)) if args.max_steps <= 0 else min(len(loader), args.max_steps)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    best, best_state, bad = float("inf"), None, 0
    val_exact = exact_crops(val_items, args) if val_items else None
    for ep in range(args.epochs):
        model.train()
        loader.dataset.epoch = ep
        t0, losses = time.time(), []
        for step, (x, yy) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            with torch.autocast("cuda", dtype=torch.float16):
                loss = masked_bce(model(x.cuda(non_blocking=True)), yy.cuda(non_blocking=True), pw, nw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
        row = {"epoch": ep, "train_loss": float(np.mean(losses)) if losses else None, "sec": round(time.time() - t0, 1),
               "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2)}
        if val_items:
            row["val_loss"] = eval_loss(model, val_items, args, pw, nw, val_exact)
        log["epochs"].append(row)
        print(f"    эпоха {ep}: {json.dumps(row, ensure_ascii=False)}", flush=True)
        if val_items:
            if row["val_loss"] < best - 1e-4:
                best, bad = row["val_loss"], 0
                names = set(model.trainable_state())
                best_state = {n: t.detach().clone() for n, t in model.state_dict().items() if n in names}
                log["best_epoch"] = ep
            else:
                bad += 1
                if bad > args.patience:
                    break
    if best_state is not None:
        sd = model.state_dict()
        model.load_state_dict({**sd, **best_state})
    return model, log


def load_folds(args: argparse.Namespace, items: list[dict[str, Any]]) -> tuple[dict[str, int], str]:
    """video/group -> фолд: JSON базы P2 (--folds-json или первый *fold*.json в r4/baseline_oof), иначе свои."""
    path = args.folds_json
    if path is None and args.baseline_dir.is_dir():
        found = sorted(args.baseline_dir.glob("*fold*.json"))
        path = found[0] if found else None
    if path is not None and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # P2: {"k", "rule", "group_fold": {группа: фолд}}; допускаем и плоский {video: fold}.
        # P2: {"k", "rule", "group_fold", "video_group", "video_fold", "source"}; video_fold — оба написания Х/X.
        folds = {}
        for key in ("group_fold", "folds", "video_fold"):
            for k, v in (raw.get(key) or {}).items():
                folds[normalize_video_id(k)] = int(v)
        if not folds:
            folds = {normalize_video_id(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}
        for k, v in (raw.get("video_fold") or {}).items():
            if folds.get(normalize_video_id(k), v) != v:
                raise ValueError(f"{path}: video_fold и group_fold расходятся для {k}")
        missing = sorted({it["group"] for it in items if folds.get(it["video"], folds.get(it["group"])) is None})
        if missing:
            print(f"нет фолда в {path} для групп {missing} — исключены", file=sys.stderr)
        return folds, str(path)
    from train_heads import assign_folds

    safe = [{"group": it["group"], "variant": it["variant"],
             "y": -1 if np.isnan(it["y"][0]) else int(it["y"][0])} for it in items]
    return assign_folds(safe, args.k), "train_heads.assign_folds (свои)"


def aliases_groups() -> dict[str, str]:
    """video_id (оба написания) -> группа, как train_heads.group_of."""
    aliases = {normalize_video_id(k): v for k, v in known_aliases().items()}
    return {k: group_of(k, aliases) for k in aliases}


def fold_of(it: dict[str, Any], folds: dict[str, int]) -> Optional[int]:
    f = folds.get(it["video"])
    return folds.get(it["group"]) if f is None else f


def summarize(items: list[dict[str, Any]], p: np.ndarray) -> dict[str, Any]:
    from train_heads import roc_auc

    out: dict[str, Any] = {}
    for k, task in enumerate(TASK_NAMES):
        for v in ALL_VARIANTS:
            idx = [i for i, it in enumerate(items) if it["variant"] == v and not np.isnan(it["y"][k])
                   and not np.isnan(p[i, k])]
            if not idx:
                continue
            yy = np.array([items[i]["y"][k] for i in idx])
            pp = p[idx, k]
            out.setdefault(task, {})[v] = {
                "auc": roc_auc(yy, pp), "acc": float(((pp >= 0.5) == yy).mean()), "n": len(idx), "pos": int(yy.sum()),
                "recall": float(((pp >= 0.5) & (yy == 1)).sum() / max(yy.sum(), 1)),
                "false_pos": int(((pp >= 0.5) & (yy == 0)).sum()),
                "max_p_neg": float(pp[yy == 0].max()) if (yy == 0).any() else None}
    # Первые 10 с persp у закрытого сейфа — место, где база давала 0,66–0,76.
    for v in ("persp", "cache_persp"):
        idx = [i for i, it in enumerate(items) if it["variant"] == v and it["t"] < 10 and it["y"][0] == 0
               and not np.isnan(p[i, 0])]
        if idx:
            out.setdefault("safe_open", {})[f"{v}_first10s_closed"] = {
                "n": len(idx), "max_p": float(p[idx, 0].max()), "mean_p": float(p[idx, 0].mean()),
                "n_ge_0.5": int((p[idx, 0] >= 0.5).sum())}
    return out


def cmd_cv(args: argparse.Namespace) -> int:
    import torch

    limit_memory(args.mem_gb)
    items = load_items(args.cache, {"safe": args.safe_labels, "desk": args.desk_labels},
                       {"safe": args.safe_label_time, "desk": args.desk_label_time})
    items = [it for it in items if any(TASKS[k][0] in args.tasks for k in np.where(~np.isnan(it["y"]))[0])
             or args.keep_unlabeled]
    for it in items:   # задачи вне --tasks не учим
        for k, t in enumerate(TASK_NAMES):
            if t not in args.tasks:
                it["y"][k] = np.nan
    if not items:
        print("нет вырезок/меток: сначала finetune_siglip.py cache", file=sys.stderr)
        return 2
    folds, folds_src = load_folds(args, items)
    for it in items:
        it["fold"] = fold_of(it, folds)
    items = [it for it in items if it["fold"] is not None]
    k_all = sorted({it["fold"] for it in items})
    run_folds = args.only_fold if args.only_fold is not None else k_all
    print(f"вырезок {len(items)}; фолды {k_all} ({folds_src}); варианты в обучении {args.train_variants}", flush=True)
    for t in args.tasks:
        k = TASK_NAMES.index(t)
        n = sum(1 for it in items if it["variant"] == "orig" and not np.isnan(it["y"][k]))
        pos = sum(1 for it in items if it["variant"] == "orig" and it["y"][k] == 1)
        print(f"  {t}: оригинал {n} размечено, {pos} положительных", flush=True)
    p = np.full((len(items), len(TASKS)), np.nan, dtype=np.float32)
    logs = []
    sample_rows: list[dict[str, Any]] = []
    label_sets = {"safe": safe_labels(args.safe_labels)} if args.safe_labels.exists() else {}
    if args.desk_labels.exists():
        label_sets["desk"] = desk_labels(args.desk_labels)
    t_start = time.time()
    for f in run_folds:
        test_idx = [i for i, it in enumerate(items) if it["fold"] == f]
        others = [g for g in k_all if g != f]
        val_fold = others[(k_all.index(f)) % len(others)] if args.val == "inner" and others else None
        train = [it for it in items if it["fold"] not in (f, val_fold) and it["variant"] in args.train_variants
                 and not np.all(np.isnan(it["y"]))]
        val = [it for it in items if it["fold"] == val_fold and it["variant"] == "orig"
               and not np.all(np.isnan(it["y"]))] if val_fold is not None else []
        if args.val == "test":
            val = [items[i] for i in test_idx if items[i]["variant"] == "orig"]
        print(f"фолд {f}: обучение {len(train)}, валидация {len(val)} (фолд {val_fold}), тест {len(test_idx)}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model, log = train_one(train, val, args, seed=args.seed + f)
        p[test_idx] = predict(model, [items[i] for i in test_idx], args)   # без потерь, из исходных кадров
        if args.samples_dir is not None:
            vids = {v for v, fv in folds.items() if fv == f} | {it["video"] for it in items if it["fold"] == f}
            vids |= {normalize_video_id(v) for v, g in aliases_groups().items() if folds.get(g) == f}
            sample_rows += predict_samples(model, vids, f, args, label_sets)
        log.update({"fold": f, "val_fold": val_fold, "n_train": len(train), "n_test": len(test_idx),
                    "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2)})
        log["test"] = summarize([items[i] for i in test_idx], p[test_idx])
        logs.append(log)
        del model
        torch.cuda.empty_cache()
    summary = summarize(items, p)
    if args.samples_dir is not None:
        args.samples_dir.mkdir(parents=True, exist_ok=True)
        save_samples(sample_rows, args.samples_dir, args.tasks)
        print(f"предсказания на кадрах state_samples: {len(sample_rows)} -> {args.samples_dir}", flush=True)
    out = args.oof_dir or args.report / args.run
    out.mkdir(parents=True, exist_ok=True)
    for k, task in enumerate(TASK_NAMES):
        if task not in args.tasks:
            continue
        oof: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for it, pr in zip(items, p[:, k]):
            if it["zone"] != TASKS[k][1] or np.isnan(pr):
                continue
            yv = it["y"][k]
            oof[it["variant"]][it["video"]].append([it["t"], round(float(pr), 4), -1 if np.isnan(yv) else int(yv)])
        (out / f"{task}_oof.json").write_text(json.dumps(oof, ensure_ascii=False), encoding="utf-8")
        # Тот же формат, что baseline_oof P2: <task>_<variant>.npz (video, t, t_ms, y, p, fold).
        for v in ALL_VARIANTS:
            idx = [i for i, it in enumerate(items) if it["variant"] == v and it["zone"] == TASKS[k][1]
                   and not np.isnan(p[i, k])]
            if not idx:
                continue
            np.savez(out / f"{task}_{v}.npz", video=np.array([items[i]["video"] for i in idx]),
                     t=np.array([items[i]["t"] for i in idx], dtype=np.float32),
                     t_ms=np.array([int(round(items[i]["t"] * 1000)) for i in idx], dtype=np.int64),
                     y=np.array([-1 if np.isnan(items[i]["y"][k]) else int(items[i]["y"][k]) for i in idx], dtype=np.int8),
                     p=p[idx, k].astype(np.float32), fold=np.array([items[i]["fold"] for i in idx], dtype=np.int8))
    desk_k = [k for k, (t, z) in enumerate(TASKS) if z == "desk" and t in args.tasks]
    if desk_k:
        # Формат P2 desk_items_oof.npz: один файл, y/p — (n, C), classes.
        idx = [i for i, it in enumerate(items) if it["zone"] == "desk" and not np.isnan(p[i, desk_k[0]])]
        yy = np.stack([items[i]["y"][desk_k] for i in idx])
        np.savez(out / "desk_items_oof.npz", variant=np.array([items[i]["variant"] for i in idx]),
                 video=np.array([items[i]["video"] for i in idx]),
                 t=np.array([items[i]["t"] for i in idx], dtype=np.float32),
                 t_ms=np.array([int(round(items[i]["t"] * 1000)) for i in idx], dtype=np.int64),
                 y=np.where(np.isnan(yy), -1, yy).astype(np.int8), p=p[np.ix_(idx, desk_k)].astype(np.float32),
                 fold=np.array([items[i]["fold"] for i in idx], dtype=np.int8),
                 classes=np.array([TASK_NAMES[k][5:] for k in desk_k]))
    groups = {}
    for it in items:
        groups[it["video"]] = it["fold"]
    (out / "folds.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {"summary": summary, "folds": logs, "folds_src": folds_src, "args": hparams(args),
              "minutes": round((time.time() - t_start) / 60, 1), "commit": git_commit()}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print_summary(summary)
    cmp = compare_baseline(out, args.baseline_dir, args.tasks, args.desk_baseline)
    if cmp:
        (out / "compare_baseline.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}  ({report['minutes']} мин)")
    return 0


def average_precision(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.sum() == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def _base_table(base_dir: Path, task: str, variant: str, desk_base: Optional[Path]) -> Optional[dict]:
    """(видео, t_ms) -> (y, p) базы P2: <task>_<variant>.npz или, для desk_*, общий desk_items_oof.npz."""
    a = base_dir / f"{task}_{variant}.npz"
    if a.exists():
        z = np.load(a)
        return {(normalize_video_id(str(vid)), int(t)): (int(y), float(p))
                for vid, t, y, p in zip(z["video"], z["t_ms"], z["y"], z["p"])}
    if task.startswith("desk_") and desk_base is not None and desk_base.exists():
        z = np.load(desk_base)
        classes = [str(c) for c in z["classes"]]
        if task[5:] not in classes:
            return None
        c = classes.index(task[5:])
        m = z["variant"] == variant
        return {(normalize_video_id(str(vid)), int(t)): (int(y[c]), float(p[c]))
                for vid, t, y, p in zip(z["video"][m], z["t_ms"][m], z["y"][m], z["p"][m])}
    return None


def compare_baseline(out: Path, base_dir: Path, tasks: list[str], desk_base: Optional[Path] = None) -> dict[str, Any]:
    """База P2 против дообучения на общих (видео, t_ms) — только фолды, что прогнаны."""
    from train_heads import roc_auc

    res: dict[str, Any] = {}
    for task in tasks:
        for v in ALL_VARIANTS:
            b = out / f"{task}_{v}.npz"
            pa = _base_table(base_dir, task, v, desk_base) if b.exists() else None
            if not pa:
                continue
            zb = np.load(b)
            rows = []
            for vid, tm, p, t in zip(zb["video"], zb["t_ms"], zb["p"], zb["t"]):
                key = (normalize_video_id(str(vid)), int(tm))
                if key in pa:
                    rows.append((pa[key], float(p), float(t)))
            rows = [r for r in rows if r[0][0] >= 0]
            if not rows:
                continue
            y = np.array([r[0][0] for r in rows])
            p_base, p_ft, t = np.array([r[0][1] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
            row = {"n": len(rows), "pos": int(y.sum())}
            for name, pp in (("base", p_base), ("ft", p_ft)):
                neg = pp[y == 0]
                row[name] = {"auc": roc_auc(y, pp), "ap": average_precision(y, pp),
                             "false_pos@0.5": int(((pp >= 0.5) & (y == 0)).sum()),
                             "neg_q95": float(np.quantile(neg, 0.95)) if len(neg) else None,
                             "neg_q99": float(np.quantile(neg, 0.99)) if len(neg) else None,
                             "recall@0.5": float(((pp >= 0.5) & (y == 1)).sum() / max(y.sum(), 1)),
                             "max_p_neg": float(pp[y == 0].max()) if (y == 0).any() else None}
                if v == "persp":
                    m = (t < 10) & (y == 0)
                    row[name]["first10s_closed_max_p"] = float(pp[m].max()) if m.any() else None
            res.setdefault(task, {})[v] = row
            b_, f_ = row["base"], row["ft"]
            print(f"  {task} {v:<12} n {row['n']:>5} (+{row['pos']})  AUC {b_['auc']:.4f} -> {f_['auc']:.4f}  "
                  f"AP {b_['ap']:.3f} -> {f_['ap']:.3f}  ложн.+@0,5 {b_['false_pos@0.5']} -> {f_['false_pos@0.5']}  "
                  f"q95(−) {b_['neg_q95']:.3f} -> {f_['neg_q95']:.3f}  q99(−) {b_['neg_q99']:.3f} -> {f_['neg_q99']:.3f}",
                  flush=True)
    return res


def print_summary(summary: dict[str, Any]) -> None:
    for task, per in summary.items():
        print(f"\n{task}:")
        for v, s in per.items():
            if "auc" in s:
                auc = "—" if s["auc"] is None else f"{s['auc']:.3f}"
                print(f"  {v:<8} AUC {auc}  точн. {s['acc']:.3f}  полнота {s['recall']:.3f}  "
                      f"ложн.+ {s['false_pos']}  max p(−) {s['max_p_neg']}  n {s['n']} (+{s['pos']})")
            else:
                print(f"  {v}: {s}")


def hparams(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("blocks", "grad_ckpt", "freeze_pool", "batch", "epochs", "lr_blocks", "lr_head", "wd", "init_l2",
            "patience", "val", "train_variants", "tasks", "seed", "max_steps", "p_clean", "safe_label_time",
            "desk_label_time", "workers")
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def cmd_final(args: argparse.Namespace) -> int:
    import torch

    limit_memory(args.mem_gb)
    items = load_items(args.cache, {"safe": args.safe_labels, "desk": args.desk_labels},
                       {"safe": args.safe_label_time, "desk": args.desk_label_time})
    for it in items:
        for k, t in enumerate(TASK_NAMES):
            if t not in args.tasks:
                it["y"][k] = np.nan
    train = [it for it in items if it["variant"] in args.train_variants and not np.all(np.isnan(it["y"]))]
    folds, folds_src = load_folds(args, items)
    epochs_src = "--epochs"
    if args.cv_report is not None:
        cv = json.loads(args.cv_report.read_text(encoding="utf-8"))
        best = [f["best_epoch"] + 1 for f in cv.get("folds", []) if f.get("best_epoch") is not None]
        if best:
            args.epochs = max(1, int(round(float(np.median(best)))))
            epochs_src = f"медиана best_epoch+1 по фолдам cv {args.cv_report}: {best}"
    print(f"final: обучение {len(train)} вырезок, {args.epochs} эпох ({epochs_src}), без валидации", flush=True)
    model, log = train_one(train, [], args, seed=args.seed)
    out = models_dir() / "ft" / "siglip_states" if args.out is None else args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.trainable_state(), out / "state.pt")
    meta = {"base": "google/siglip-base-patch16-224", "base_dir": "models/hf/siglip-base-patch16-224",
            "tasks": [{"name": t, "zone": z, "index": i} for i, (t, z) in enumerate(TASKS) if t in args.tasks],
            "all_tasks": list(TASK_NAMES), "trainable_blocks": args.blocks, "first_trainable_layer": 12 - args.blocks,
            "train_pool": not args.freeze_pool, "zone_crop_margin": ZONE_CROP_MARGIN,
            "preprocess": "RGB, PIL bicubic 224x224, x/255, (x-0.5)/0.5 (SiglipImageProcessor)",
            "embedding": "head(post_layernorm(h)), L2; p = sigmoid(emb @ cls.weight.T + cls.bias)",
            "labels": {"safe": str(args.safe_labels), "desk": str(args.desk_labels) if args.desk_labels.exists() else None},
            "n_train": len(train), "folds": {k: v for k, v in folds.items()}, "folds_src": folds_src,
            "epochs": args.epochs, "epochs_src": epochs_src,
            "label_time": {"safe": args.safe_label_time, "desk": args.desk_label_time},
            "hparams": hparams(args), "train_log": log, "commit": git_commit(),
            "date": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    size = (out / "state.pt").stat().st_size / 1e6
    print(f"-> {out}/state.pt ({size:.0f} МБ), meta.json")
    return 0


def load_finetuned(state_dir: Path, device: str = "cuda") -> tuple[Any, dict[str, Any]]:
    """Дообученная башня для инференса: база из models/hf + state.pt поверх."""
    import torch

    meta = json.loads((state_dir / "meta.json").read_text(encoding="utf-8"))
    model = build_model(int(meta["trainable_blocks"]), False, device, bool(meta.get("train_pool", True)))
    sd = model.state_dict()
    state = torch.load(state_dir / "state.pt", map_location=device)
    model.load_state_dict({**sd, **{n: t.to(sd[n].dtype) for n, t in state.items()}})
    if device == "cpu":
        model.float()      # fp16-свёртки/матмулы на CPU без autocast не поддерживаются или медленны
    return model.eval(), meta


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    r4 = ROOT / "data" / "ml_results" / "r4"      # фолды и OOF базы раунда 4
    p = argparse.ArgumentParser(description="Частичное дообучение визуальной башни SigLIP (safe_open + desk_items).")
    p.add_argument("cmd", choices=("bench", "cache", "cv", "final"))
    p.add_argument("--mem-gb", type=float, default=1.8, help="лимит аллокатора torch на GPU, ГБ")
    p.add_argument("--blocks", type=int, nargs="+", default=[4], help="число дообучаемых блоков (bench — список)")
    p.add_argument("--ckpt", choices=("off", "on", "both"), default="both", help="bench: gradient checkpointing")
    p.add_argument("--grad-ckpt", action="store_true", help="cv/final: gradient checkpointing")
    p.add_argument("--freeze-pool", action="store_true", help="не обучать голову attention pooling")
    p.add_argument("--batch", type=int, nargs="+", default=[16])
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=10, help="bench: шагов замера")
    p.add_argument("--out", type=Path, default=None)
    # данные
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw_cache_r3")
    p.add_argument("--frames", type=Path, default=ROOT / "data" / "frames_r2")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "ft_crops")
    p.add_argument("--emb", type=Path, default=ROOT / "data" / "emb", help="замороженные эмбеддинги для тёплого старта")
    p.add_argument("--zones", nargs="+", default=["safe", "desk"])
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=ALL_VARIANTS)
    p.add_argument("--only", action="append")
    p.add_argument("--every", type=float, default=1.0)
    p.add_argument("--ctx-margin", type=float, default=0.15, help="запас рамки ctx под онлайн-сдвиг, доля")
    p.add_argument("--ctx-size", type=int, default=256)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--max-mb", type=float, default=1500)
    p.add_argument("--safe-labels", type=Path, default=ROOT / "training" / "labels" / "safe_open.csv")
    p.add_argument("--desk-labels", type=Path, default=ROOT / "training" / "labels" / "desk_items.csv")
    p.add_argument("--tasks", nargs="+", default=list(TASK_NAMES), choices=TASK_NAMES)
    p.add_argument("--safe-label-time", choices=("exact", "round"), default="exact",
                   help="метка safe по точному t (как база P2 safe_open) или round(t)")
    p.add_argument("--desk-label-time", choices=("exact", "round"), default="round",
                   help="метка desk по round(t) (как P2 desk_items)")
    p.add_argument("--p-clean", type=float, default=0.3, help="доля обучающих вырезок без аугментаций")
    p.add_argument("--cv-report", type=Path, default=None,
                   help="final: report.json прогона cv — эпох = медиана (best_epoch + 1) по фолдам")
    p.add_argument("--keep-unlabeled", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--folds-json", type=Path, default=None, help="video->fold базы P2")
    p.add_argument("--baseline-dir", type=Path, default=r4 / "baseline_oof")
    p.add_argument("--desk-baseline", type=Path, default=r4 / "p2" / "desk_items_audited" / "desk_items_oof.npz")
    p.add_argument("--k", type=int, default=5, help="число фолдов, если своих")
    p.add_argument("--only-fold", type=int, nargs="+", default=None)
    p.add_argument("--val", choices=("inner", "test", "none"), default="inner",
                   help="ранняя остановка: inner — следующий обучающий фолд; test — тестовый (оптимистично)")
    p.add_argument("--train-variants", nargs="+", default=list(VARIANTS), choices=ALL_VARIANTS,
                   help="cache_* по умолчанию только для оценки")
    # обучение
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="ограничить шаги эпохи (smoke)")
    p.add_argument("--lr-blocks", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--init-l2", type=float, default=1e-5, help="L2 fit_logreg для тёплого старта головы")
    p.add_argument("--workers", type=int, default=2, help="процессов DataLoader (RAM хоста делим с prod-worker)")
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--run", default=None, help="имя прогона cv")
    p.add_argument("--report", type=Path, default=ROOT / "data" / "ml_results" / "r4" / "ft")
    p.add_argument("--oof-dir", type=Path, default=None, help="cv: каталог вывода целиком (вместо --report/--run)")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="cv: ещё и предсказания модели фолда на кадрах state_samples сырых кэшей (orig + cache_*)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.cmd in ("cv", "final"):
        args.blocks = args.blocks[0]
        args.batch = args.batch[0]
        if args.run is None:
            args.run = f"b{args.blocks}_{time.strftime('%H%M')}"
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd in ("bench", "cv", "final"):
        import torch

        if not torch.cuda.is_available():
            print("нужна видеокарта", file=sys.stderr)
            return 2
    return {"bench": cmd_bench, "cache": cmd_cache, "cv": cmd_cv, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
