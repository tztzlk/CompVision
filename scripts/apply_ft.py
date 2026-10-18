#!/usr/bin/env python3
"""Подставить поля дообученной башни (SceneConfig.safe_ft): safe_open и desk_cash — в копию сырого кэша.

Без GPU — вероятности вне фолда из scripts/finetune_siglip.py cv (<task>_<вариант>.npz: video, t, t_ms, y, p,
fold): для raw_cache_r3 — <task>_orig.npz, для raw_cache_aug_<v> — <task>_cache_<v>.npz (кадры и зоны
аугментированного кэша, как их видит конвейер).

    PYTHONPATH=. .venv/bin/python scripts/apply_ft.py --raw data/raw_cache_r3 --out <scratch>/raw_base_ft \
        --oof <ft_oof>/safe_open_orig.npz --desk-oof <ft_oof>/desk_cash_orig.npz
    PYTHONPATH=. .venv/bin/python scripts/rebuild_from_raw.py --raw <scratch>/raw_base_ft --out <scratch>/scene_base_ft --code-config

safe_open (--oof): сопоставление по ключу t в мс (время в .npz — float32; допуск --tol-ms, 1 мс). Секунды
без вероятности в OOF (cv оценивает только размеченные open/closed — occluded нет) — None либо, с --fill-raw
<каталог сырого кэша>, значение safe_open оттуда, либо, с --keep-missing, прежнее значение входного кэша
(вход — кэш с головой раунда 3 вне фолда из apply_head.py: как сервис, где башня видит и заслонённый сейф).
Прежнее значение уходит в state_samples.safe_open_zero_shot (кэш без головы) или safe_open_head (с головой).

desk_cash (--desk-oof): ряд state_samples.desk_cash на секундах выборок стола сырого кэша (cash_on_desk, раз
в state_every_sec — те же, что у конвейера с флагом); значение — OOF с ближайшим ключом t_ms в пределах
--desk-tol-ms (500 мс: OOF стола с шагом 1 с; на деле времена совпадают с выборками до мс). Секунды без
OOF («?» в desk_items.csv — cv их не оценивает) — None: в окне R14 они не дают ни «есть», ни «нет».
builder.with_ft_items ставит ряд в отдельное поле сцены desk_cash_ft (только для R14), desk_cash OWLv2 не меняется.

С GPU (--model <каталог весов>) — вероятности финальной башни на кадрах (как в сервисе) по всем её задачам:
кадры data/frames_r2/<видео> (--variant orig) или рабочие каталоги data/scene_work_aug_<v> (--variant
cache_<v>), зоны — из сырого кэша, вырезка — crop_zone с ZONE_CROP_MARGIN без закрашивания людей.
Под --gpu-lock (flock) и с лимитом аллокатора --mem-frac (не больше 0,3).

meta.safe_ft (name, fields, счётчики) и config.safe_ft отмечают, что сцена считалась с башней (детекторы
R10 и R14 берут ft_* для полей из meta.safe_ft.fields).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cache import normalize_video_id  # noqa: E402

FT_NAME = "siglip_states"
FT_REL = "ft/siglip_states"
ZONE_OF = {"safe_open": "safe", "desk_cash": "desk"}
DESK_SERIES = "cash_on_desk"          # выборки стола в сыром кэше (zero-shot раз в state_every_sec)


def ms(t: float) -> int:
    return int(round(float(t) * 1000))


def load_oof(path: Path) -> dict[str, dict[int, float]]:
    """OOF .npz -> {нормализованный video_id: {t_ms: p}}."""
    z = np.load(path, allow_pickle=False)
    t_ms = z["t_ms"] if "t_ms" in z.files else np.array([ms(t) for t in z["t"]], dtype=np.int64)
    out: dict[str, dict[int, float]] = {}
    for vid, key, p in zip(z["video"], t_ms, z["p"]):
        if np.isfinite(p):
            out.setdefault(normalize_video_id(str(vid)), {})[int(key)] = float(p)
    return out


def lookup(by_ms: Mapping[int, float], t: float, tol_ms: int) -> Optional[float]:
    """Значение с ближайшим ключом в пределах tol_ms."""
    key = ms(t)
    if key in by_ms:
        return by_ms[key]
    if tol_ms <= 50:
        for d in sorted(range(-tol_ms, tol_ms + 1), key=abs):
            if key + d in by_ms:
                return by_ms[key + d]
        return None
    best = min(by_ms, key=lambda k: abs(k - key), default=None)
    return by_ms[best] if best is not None and abs(best - key) <= tol_ms else None


def _mark(raw: dict[str, Any], field: str, counts: Mapping[str, int], meta: Optional[Mapping[str, Any]]) -> None:
    ft = dict((raw.get("meta") or {}).get("safe_ft") or {"name": FT_NAME, **dict(meta or {}), "fields": []})
    ft["fields"] = list(dict.fromkeys([*ft.get("fields", []), field]))
    ft[field] = dict(counts)
    raw["meta"] = {**(raw.get("meta") or {}), "safe_ft": ft}
    raw["config"] = {**(raw.get("config") or {}), "safe_ft": FT_REL}


def substitute(raw: dict[str, Any], probs: Mapping[int, float], *, tol_ms: int = 1,
               fill: Optional[Mapping[int, Optional[float]]] = None, keep_missing: bool = False,
               meta: Optional[Mapping[str, Any]] = None) -> dict[str, int]:
    """safe_open := вероятность башни по ключу t в мс (на месте). Секунды без неё — из fill ({t_ms: p}),
    с keep_missing — прежнее значение, иначе None. Нет ряда safe_open (нет зоны сейфа) или ни одного
    значения — без изменений. Возвращает счётчики {replaced, filled, missing}."""
    samples = raw.get("state_samples") or {}
    current = samples.get("safe_open")
    counts = {"replaced": 0, "filled": 0, "missing": 0}
    if not current:
        return counts
    fill = {k: v for k, v in (fill or {}).items() if v is not None}
    if keep_missing:
        fill = {**{ms(t): p for t, p in current if p is not None}, **fill}
    new = []
    for t, _ in current:
        p = lookup(probs, t, tol_ms)
        if p is not None:
            counts["replaced"] += 1
        elif (p := lookup(fill, t, tol_ms)) is not None:
            counts["filled"] += 1
        else:
            counts["missing"] += 1
        new.append([t, None if p is None else round(float(p), 4)])
    if not counts["replaced"] and not counts["filled"]:
        return counts                     # видео нет в OOF — не трогаем (сцена как без башни)
    keep = "safe_open_head" if "safe_open_zero_shot" in samples else "safe_open_zero_shot"
    samples[keep] = current
    samples["safe_open"] = new
    _mark(raw, "safe_open", counts, meta)
    return counts


def substitute_desk(raw: dict[str, Any], probs: Mapping[int, float], *, tol_ms: int = 500,
                    meta: Optional[Mapping[str, Any]] = None) -> dict[str, int]:
    """state_samples.desk_cash := вероятность башни на секундах выборок стола (ближайший ключ в пределах
    tol_ms; нет — None). Нет выборок стола или ни одного значения — без изменений."""
    samples = raw.get("state_samples") or {}
    times = [t for t, _ in samples.get(DESK_SERIES) or []]
    counts = {"replaced": 0, "missing": 0}
    if not times or not probs:
        return counts
    series = []
    for t in times:
        p = lookup(probs, t, tol_ms)
        counts["replaced" if p is not None else "missing"] += 1
        series.append([t, None if p is None else round(float(p), 4)])
    if not counts["replaced"]:
        return counts
    samples["desk_cash"] = series
    _mark(raw, "desk_cash", counts, meta)
    return counts


@contextlib.contextmanager
def gpu_lock(path: Optional[Path]) -> Iterator[None]:
    if path is None:
        yield
        return
    import fcntl

    with open(path, "a") as fh:
        print(f"жду {path} ...", flush=True)
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def model_probs(raw: dict[str, Any], tower: Any, task: str, variant: str, frames_root: Path) -> dict[int, float]:
    """Вероятности финальной башни по задаче на кадрах выборок зоны сырого кэша: {t_ms: p}."""
    from finetune_siglip import _frames_dir

    from worker.perception.states import ZONE_CROP_MARGIN, crop_zone
    from worker.perception.video import imread_unicode

    zone_name = ZONE_OF[task]
    zone = raw["zones"].get(zone_name)
    series = (raw.get("state_samples") or {}).get("safe_open" if task == "safe_open" else DESK_SERIES) or []
    if zone is None or not series or task not in tower.tasks:
        return {}
    fdir, work = _frames_dir(frames_root, variant, raw["video_id"])
    paths = {ms(t): (fdir / rel if work else fdir / f"{int(idx):06d}.jpg") for idx, t, rel in raw["frames"]}
    times, crops = [], []
    for t, _ in series:
        path = paths.get(ms(t))
        image = imread_unicode(path) if path is not None else None
        crop = crop_zone(image, zone, ZONE_CROP_MARGIN.get(zone_name, 0.1)) if image is not None else None
        if crop is not None:
            times.append(t)
            crops.append(crop)
    if not crops:
        return {}
    p = tower(crops, [task])[task]
    return {ms(t): float(v) for t, v in zip(times, p)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Поля дообученной башни (safe_open, desk_cash) в копии сырого кэша.")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oof", type=Path, help="OOF safe_open .npz finetune_siglip.py cv (video, t_ms, p) — без GPU")
    ap.add_argument("--desk-oof", type=Path, help="OOF desk_cash .npz (шаг 1 с) — без GPU")
    ap.add_argument("--model", type=Path, help="каталог весов башни (state.pt, meta.json) — инференс на кадрах")
    ap.add_argument("--fill-raw", type=Path, default=None,
                    help="сырой кэш, откуда брать safe_open для секунд без вероятности в OOF")
    ap.add_argument("--keep-missing", action="store_true",
                    help="секунды без вероятности в OOF — прежнее значение safe_open (вход — кэш с головой)")
    ap.add_argument("--tol-ms", type=int, default=1)
    ap.add_argument("--desk-tol-ms", type=int, default=500)
    ap.add_argument("--variant", default="orig", help="с --model: orig | cache_<flip|persp|crop|light>")
    ap.add_argument("--frames", type=Path, default=ROOT / "data" / "frames_r2")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-frac", type=float, default=0.3, help="лимит аллокатора torch, доля памяти GPU")
    ap.add_argument("--gpu-lock", type=Path, default=None, help="файл flock для GPU")
    args = ap.parse_args(argv)
    if args.model is not None and (args.oof or args.desk_oof):
        ap.error("--model не сочетается с --oof/--desk-oof")
    if args.model is None and not (args.oof or args.desk_oof):
        ap.error("нужен --oof и/или --desk-oof, либо --model")

    files = sorted(p for p in args.raw.glob("*.json") if not p.name.startswith("_"))
    if not files:
        print(f"в {args.raw} нет сырого кэша *.json", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {"safe_open": {"replaced": 0, "filled": 0, "missing": 0}, "desk_cash": {"replaced": 0, "missing": 0}}
    n_vid = {"safe_open": 0, "desk_cash": 0}
    untouched: list[str] = []
    with contextlib.ExitStack() as stack:
        tower = None
        safe_oof = load_oof(args.oof) if args.oof else None
        desk_oof = load_oof(args.desk_oof) if args.desk_oof else None
        if args.model is None:
            meta: dict[str, Any] = {"oof": True, "source": ", ".join(p.name for p in (args.oof, args.desk_oof) if p)}
        else:
            stack.enter_context(gpu_lock(args.gpu_lock))
            from worker.perception.ft_tower import load_ft_tower

            if args.device == "cuda":
                import torch

                torch.cuda.set_per_process_memory_fraction(min(args.mem_frac, 0.3), 0)
            tower, err = load_ft_tower(args.models_dir, str(args.model.resolve()), args.device)
            if tower is None:
                print(f"башня не загружена: {err}", file=sys.stderr)
                return 2
            meta = {"oof": False, "source": str(args.model)}
        for path in files:
            raw = json.loads(path.read_text(encoding="utf-8"))
            vid = normalize_video_id(raw["video_id"])
            if tower is not None:
                safe: Optional[dict[int, float]] = model_probs(raw, tower, "safe_open", args.variant, args.frames)
                desk: Optional[dict[int, float]] = model_probs(raw, tower, "desk_cash", args.variant, args.frames)
            else:
                safe = safe_oof.get(vid, {}) if safe_oof is not None else None
                desk = desk_oof.get(vid, {}) if desk_oof is not None else None
            if safe is not None:
                fill = None
                if args.fill_raw is not None and (args.fill_raw / path.name).exists():
                    other = json.loads((args.fill_raw / path.name).read_text(encoding="utf-8"))
                    fill = {ms(t): p for t, p in (other.get("state_samples") or {}).get("safe_open") or []}
                counts = substitute(raw, safe, tol_ms=args.tol_ms, fill=fill, keep_missing=args.keep_missing,
                                    meta=meta)
                if not counts["replaced"] and (raw.get("state_samples") or {}).get("safe_open"):
                    untouched.append(raw["video_id"])
                n_vid["safe_open"] += "safe_open" in ((raw.get("meta") or {}).get("safe_ft") or {}).get("fields", [])
                for k, v in counts.items():
                    totals["safe_open"][k] += v
            if desk is not None:
                counts = substitute_desk(raw, desk, tol_ms=args.desk_tol_ms, meta=meta)
                n_vid["desk_cash"] += bool(counts["replaced"])
                for k, v in counts.items():
                    totals["desk_cash"][k] += v
            (args.out / path.name).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    s, d = totals["safe_open"], totals["desk_cash"]
    print(f"safe_open: {n_vid['safe_open']} из {len(files)} видео, выборок {s['replaced']}, дополнено {s['filled']}, "
          f"без значения {s['missing']}; desk_cash: {n_vid['desk_cash']} видео, выборок {d['replaced']}, "
          f"без значения {d['missing']} -> {args.out}")
    if untouched:
        print(f"нет вероятностей safe_open при ряде safe_open ({len(untouched)}; ряд — из дополнения или прежний): "
              f"{', '.join(untouched)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
