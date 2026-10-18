#!/usr/bin/env python3
"""Обучение лёгкой головы (логистическая регрессия) на эмбеддингах SigLIP вырезок зоны.

    PYTHONPATH=. .venv/bin/python scripts/train_heads.py safe_open
    PYTHONPATH=. .venv/bin/python scripts/train_heads.py safe_open --train-variants orig flip --l2 1e-3

Данные: data/emb/<zone>/<вариант>/<video_id>.npz (scripts/embed_zones.py) и метки
training/labels/<head>.csv (video_id,t_from,t_to,state; строки видео применяются по порядку,
следующая перекрывает предыдущую; state occluded — не учим).

Проверка — только с разбиением по видео: K фолдов по группам видео (дубликаты Х-4≡Х-47, Х-5≡Х-48 —
одна группа), все аугментированные копии видео — в фолде оригинала. Соседние кадры одного ролика почти
одинаковы, разбиение по кадрам дало бы фальшивые 99 %. Отчёт — ROC-AUC и точность при пороге 0,5 по
фолдам, отдельно на оригиналах и на каждом варианте; для сравнения — AUC zero-shot SigLIP (state_samples
сырого кэша) на тех же кадрах оригинала.

Выход: models/heads/<head>.npz (w 768, b, классы, метаданные) — голова на всех данных;
data/ml_results/r3/heads/<head>_oof.json — вероятности вне фолда по видео и t (для подбора порогов
правил по фолдам) и <head>_report.json.
Обучение — torch (L-BFGS, L2, веса классов обратно частоте); инференс головы — numpy:
sigmoid(emb @ w + b).

desk_items — мультиметочная голова предметов на столе (cash, bag, coat, cup; метки
training/labels/desk_items.csv, столбцы классов 1 / 0 / ?): по классу своя логрегрессия, «?» не учим;
w — C×768, b — C. Фолды по умолчанию — те же, что у safe_open (--folds-from). Оценка — класс × вариант:
orig и синтетические аугментации (data/emb/desk/<v>) плюс кадры аугментированных кэшей
(data/emb_aug/<v>/desk/orig, варианты cache_<v>; в обучение не входят), рядом — AUC OWLv2-скана
(desk_cash …) и zero-shot по сценам на тех же кадрах:

    PYTHONPATH=. .venv/bin/python scripts/train_heads.py desk_items        # l2 1e-5, классы cash bag
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cache import known_aliases, normalize_video_id  # noqa: E402
from worker.perception.heads import PERSON_GEO_SCALE as GEO_SCALE  # noqa: E402

HEADS: dict[str, dict[str, Any]] = {
    "safe_open": {"zone": "safe", "positive": "open", "negative": "closed", "zero_shot": "safe_open"},
    # Вырезки людей по трекам (scripts/embed_people.py); метки по (видео, трек, t); к эмбеддингу
    # добавляется геометрия рамки worker.perception.heads.person_geo по сырому кэшу, умноженная на GEO_SCALE.
    "person_inside": {"zone": "person", "positive": "1", "negative": "0", "zero_shot": None, "by_track": True},
    # Предметы на столе: вырезка зоны desk, мультиметки (training/labels/desk_items.csv: столбцы классов
    # 1 / 0 / ? — по классу своя логрегрессия, ? и пусто не учим). Листы разметки — шаг 1 с с подписью
    # round(t), эмбеддинги — все кадры 3 к/с: метка кадра — метка ближайшей секунды (round(t)).
    # compare — поля состояний сцены (scene_cache) для сравнения AUC на тех же кадрах: OWLv2-скан и zero-shot.
    # train_classes — что учим по умолчанию: у coat и cup по одному видео с предметом — фолдами не проверить.
    "desk_items": {"zone": "desk", "classes": ("cash", "bag", "coat", "cup"), "train_classes": ("cash", "bag"),
                   "zero_shot": None,
                   "compare": {"cash": ("desk_cash", "cash_on_desk"), "bag": ("desk_collection_bag", "bag_on_desk"),
                               "coat": ("desk_coat", "outerwear_on_desk"), "cup": ("desk_cup", "cup_or_food")}},
}


def person_geo_lookup(raw_dir: Path) -> dict[tuple[str, int, int], list[float]]:
    """(video_id, track_id, t в мс) -> heads.person_geo по сырому кэшу — та же геометрия, что в конвейере
    (рамки исходных треков; у аугментаций геометрия берётся от оригинала, как видит её builder)."""
    import json as _json

    from worker.perception.heads import person_geo
    from worker.perception.tracking import Track, TrackPoint
    from worker.scene.builder import track_height

    out: dict[tuple[str, int, float], list[float]] = {}
    for path in Path(raw_dir).glob("*.json"):
        raw = _json.loads(path.read_text(encoding="utf-8"))
        tracks = [Track(int(t["id"]), [TrackPoint(int(a), float(b), tuple(c), float(d)) for a, b, c, d in t["points"]])
                  for t in raw["tracks"]]
        ref, desk, vid = track_height(tracks), raw["zones"].get("desk"), normalize_video_id(raw["video_id"])
        for t in raw["tracks"]:
            for _fi, tt, box, sc in t["points"]:
                out[(vid, int(t["id"]), int(round(float(tt) * 1000)))] = person_geo(box, sc, ref, desk)
    return out
VARIANTS = ("orig", "flip", "persp", "crop", "light")


def load_labels(path: Path) -> dict[Any, list[tuple[float, float, str]]]:
    """Ключ — video_id или (video_id, track_id), если в файле есть столбец track_id."""
    rows: dict[Any, list[tuple[float, float, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        vid = normalize_video_id(r["video_id"])
        key = (vid, int(r["track_id"])) if "track_id" in r else vid
        rows[key].append((float(r["t_from"]), float(r["t_to"]), (r.get("state") or r.get("inside") or "").strip()))
    return rows


def load_multilabels(path: Path, classes: tuple[str, ...]) -> dict[str, list[tuple[float, float, dict[str, int]]]]:
    """video_id -> [(t_from, t_to, {класс: 1 | 0 | -1})]; -1 — «?» или пусто (не учим). Строка перекрывает
    предыдущие целиком (все классы)."""
    rows: dict[str, list[tuple[float, float, dict[str, int]]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        if not (r.get("video_id") or "").strip() or r.get("t_from") in (None, "") or r.get("t_to") in (None, ""):
            continue
        vals = {c: 1 if (r.get(c) or "").strip() == "1" else 0 if (r.get(c) or "").strip() == "0" else -1
                for c in classes}
        rows[normalize_video_id(r["video_id"].strip())].append((float(r["t_from"]), float(r["t_to"]), vals))
    return rows


def average_precision(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def calibration(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Распределение вероятностей: квантили на положительных (5/25/50 %) и отрицательных (50/95/99 %, max)."""
    pos, neg = p[y == 1], p[y == 0]

    def q(a: np.ndarray, qs: list[float]) -> Optional[list[float]]:
        return [round(float(v), 3) for v in np.quantile(a, qs)] if len(a) else None

    return {"pos_q05_25_50": q(pos, [0.05, 0.25, 0.5]), "neg_q50_95_99_max": q(neg, [0.5, 0.95, 0.99, 1.0]),
            "recall@0.5": float((pos >= 0.5).mean()) if len(pos) else None,
            "fp@0.5": int((neg >= 0.5).sum()), "fp@0.8": int((neg >= 0.8).sum())}


def load_folds(path: Path) -> dict[str, int]:
    """Группа -> фолд из *_folds.npz (ключ groups) или JSON (group_fold или плоский словарь)."""
    if path.suffix == ".npz":
        return {k: int(v) for k, v in json.loads(str(np.load(path)["groups"])).items()}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in (data.get("group_fold") or data).items()}


def label_at(segments: list[tuple[float, float, Any]], t: float) -> Optional[Any]:
    state = None
    for t0, t1, s in segments:
        if t0 - 1e-6 <= t <= t1 + 1e-6:
            state = s
    return state


def group_of(vid: str, aliases: dict[str, tuple[str, ...]]) -> str:
    ids = aliases.get(vid) or aliases.get(vid.replace("X", "Х")) or (vid,)
    return min(normalize_video_id(i) for i in ids)


def roc_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allp = np.concatenate([pos, neg])[order]
    # средние ранги для равных значений
    i = 0
    while i < len(allp):
        j = i
        while j + 1 < len(allp) and allp[j + 1] == allp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float, iters: int = 200) -> tuple[np.ndarray, float]:
    import torch

    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    weights = torch.where(yt > 0.5, len(y) / (2 * max(n_pos, 1)), len(y) / (2 * max(n_neg, 1)))
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # Эмбеддинги нормированы (|e| = 1), логиты малы — масштаб 10 ускоряет сходимость.
    scale = 10.0
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure() -> Any:
        opt.zero_grad()
        logits = scale * (xt @ w) + b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, weight=weights)
                + l2 * (scale * w).pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return (scale * w).detach().numpy().astype(np.float32), float(b.detach())


def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x @ w + b)))


def load_data(head: str, emb_root: Path, labels: dict[str, list[tuple[float, float, str]]],
              aliases: dict[str, tuple[str, ...]], raw_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    spec = HEADS[head]
    items = []
    geo_by = person_geo_lookup(raw_dir) if spec.get("by_track") and raw_dir is not None else {}
    scale = np.asarray(GEO_SCALE, dtype=np.float32)
    for variant in VARIANTS:
        vdir = emb_root / spec["zone"] / variant
        for path in sorted(vdir.glob("*.npz")):
            vid = normalize_video_id(path.stem)
            segs = labels.get(vid)
            if not segs and not spec.get("by_track"):
                continue
            data = np.load(path)
            if spec.get("by_track"):
                for tr, t, e in zip(data["track"], data["t"], data["emb"]):
                    g = geo_by.get((vid, int(tr), int(round(float(t) * 1000))))
                    if g is None:
                        continue
                    g = np.asarray(g, dtype=np.float32) * scale
                    s = label_at(labels.get((vid, int(tr)), []), float(t))
                    y = 1 if s == spec["positive"] else 0 if s == spec["negative"] else -1
                    items.append({"variant": variant, "video": vid, "group": group_of(vid, aliases), "t": float(t),
                                  "track": int(tr), "y": y, "emb": np.concatenate([e, g])})
                continue
            for t, e in zip(data["t"], data["emb"]):
                s = label_at(segs, float(t))
                y = 1 if s == spec["positive"] else 0 if s == spec["negative"] else -1   # -1: не учим, но предсказываем
                items.append({"variant": variant, "video": vid, "group": group_of(vid, aliases), "t": float(t),
                              "y": y, "emb": e})
    return items


def assign_folds(items: list[dict[str, Any]], k: int) -> dict[str, int]:
    """Группы видео по фолдам: сначала группы с открытым сейфом (по кругу), затем остальные."""
    pos_groups = sorted({it["group"] for it in items if it["y"] == 1 and it["variant"] == "orig"})
    # Ровнее по фолдам: группы с открытым сейфом — по убыванию числа открытых кадров.
    n_pos = defaultdict(int)
    for it in items:
        if it["y"] == 1 and it["variant"] == "orig":
            n_pos[it["group"]] += 1
    pos_groups.sort(key=lambda g: (-n_pos[g], g))
    all_groups = sorted({it["group"] for it in items})
    rest = [g for g in all_groups if g not in pos_groups]
    folds: dict[str, int] = {}
    for i, g in enumerate(pos_groups + rest):
        folds[g] = i % k
    return folds


def zero_shot_lookup(raw_dir: Path, key: str) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {}
    for path in raw_dir.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        out[normalize_video_id(raw["video_id"])] = {int(round(float(t) * 1000)): p for t, p in raw["state_samples"].get(key, [])
                                                    if p is not None}
    return out


def scene_values(scene_dir: Path, fields: list[str]) -> dict[tuple[str, int], dict[str, float]]:
    """(video_id, t в мс) -> значения полей состояний готовой сцены (для сравнения с головой)."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    if not scene_dir.is_dir():
        return out
    for path in scene_dir.glob("*.json"):
        if path.name.startswith("_"):
            continue
        scene = json.loads(path.read_text(encoding="utf-8"))
        vid = normalize_video_id(scene.get("video_id") or path.stem)
        for s in scene.get("states", []):
            out[(vid, int(round(float(s["t"]) * 1000)))] = {f: s.get(f) for f in fields}
    return out


def load_multi_data(spec: dict[str, Any], sources: list[tuple[str, Path]],
                    labels: dict[str, list[tuple[float, float, dict[str, int]]]],
                    aliases: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    """Вырезки размеченных видео: y — по классу 1 / 0 / -1 (не учим); метка кадра — по round(t) (листы — шаг 1 с)."""
    classes = spec["classes"]
    items = []
    for name, vdir in sources:
        for path in sorted(vdir.glob("*.npz")):
            vid = normalize_video_id(path.stem)
            segs = labels.get(vid)
            if not segs:
                continue
            data = np.load(path)
            t_ms = data["t_ms"] if "t_ms" in data else np.round(data["t"].astype(np.float64) * 1000).astype(np.int64)
            for t, ms, e in zip(data["t"], t_ms, data["emb"].astype(np.float32)):
                lab = label_at(segs, float(round(float(t))))
                items.append({"variant": name, "video": vid, "group": group_of(vid, aliases), "t": float(t),
                              "t_ms": int(ms), "y": [-1 if lab is None else lab[c] for c in classes], "emb": e})
    return items


def train_multilabel(args: argparse.Namespace, spec: dict[str, Any], aliases: dict[str, tuple[str, ...]]) -> int:
    """Мультиметочная голова: по классу своя логрегрессия на тех же фолдах; отчёт — класс × вариант."""
    classes = tuple(args.classes or spec.get("train_classes") or spec["classes"])
    labels = load_multilabels(args.labels or ROOT / "training" / "labels" / f"{args.head}.csv", classes)
    aug = tuple(v for v in VARIANTS if v != "orig")
    sources = [(v, args.emb / spec["zone"] / v) for v in VARIANTS]
    sources += [(f"cache_{v}", args.aug_emb / v / spec["zone"] / "orig") for v in aug]
    items = load_multi_data({**spec, "classes": classes}, sources, labels, aliases)
    if not items:
        print("нет данных: сначала эмбеддинги зоны и метки", file=sys.stderr)
        return 2
    folds_path = args.folds_from or ROOT / "data" / "ml_results" / "r3" / "heads" / "safe_open_folds.npz"
    if folds_path.exists():
        folds = load_folds(folds_path)
    else:
        print(f"нет {folds_path} — фолды заново по группам", file=sys.stderr)
        folds = {g: i % args.folds for i, g in enumerate(sorted({it["group"] for it in items}))}
        folds_path = None
    missing = sorted({it["group"] for it in items} - set(folds))
    if missing:
        print(f"группы без фолда: {', '.join(missing)}", file=sys.stderr)
        return 2
    k = max(folds.values()) + 1
    x = np.stack([it["emb"] for it in items])
    y = np.array([it["y"] for it in items], dtype=np.int8)            # N × C
    var = np.array([it["variant"] for it in items])
    fold = np.array([folds[it["group"]] for it in items])
    variants = [name for name, _ in sources if (var == name).any()]
    train_var = np.isin(var, args.train_variants)

    print(f"[{args.head}] видео {len({it['video'] for it in items})}, фолдов {k}; по вариантам — вырезок / положительных:")
    for v in variants:
        m = var == v
        print(f"  {v:<12}{int(m.sum()):>6}  " + "  ".join(f"{c} {int((y[m, i] == 1).sum())}/{int((y[m, i] >= 0).sum())}"
                                                        for i, c in enumerate(classes)))

    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_w = np.zeros((k, len(classes), x.shape[1]), dtype=np.float32)
    fold_b = np.zeros((k, len(classes)), dtype=np.float32)
    final_w = np.zeros((len(classes), x.shape[1]), dtype=np.float32)
    final_b = np.zeros(len(classes), dtype=np.float32)
    for ci in range(len(classes)):
        lab = y[:, ci] >= 0
        for f in range(k):
            tr = (fold != f) & train_var & lab
            te = fold == f
            if (y[tr, ci] == 1).sum() == 0 or (y[tr, ci] == 0).sum() == 0:
                continue
            fold_w[f, ci], fold_b[f, ci] = fit_logreg(x[tr], y[tr, ci], args.l2)
            oof[te, ci] = predict(x[te], fold_w[f, ci], fold_b[f, ci])
        tr = train_var & lab
        if (y[tr, ci] == 1).any() and (y[tr, ci] == 0).any():
            final_w[ci], final_b[ci] = fit_logreg(x[tr], y[tr, ci], args.l2)

    # Сравнение: те же кадры, поля сцены (OWLv2-скан стола и zero-shot SigLIP).
    compare = spec.get("compare") or {}
    fields = sorted({f for fs in compare.values() for f in fs})
    scene_dirs = {"orig": args.scene, **{f"cache_{v}": args.scene.parent / f"scene_cache_aug_{v}" for v in aug}}
    scene = {name: scene_values(d, fields) for name, d in scene_dirs.items() if name in variants}

    summary: dict[str, dict[str, Any]] = {}
    for ci, c in enumerate(classes):
        summary[c] = {}
        for v in variants:
            m = (var == v) & (y[:, ci] >= 0) & ~np.isnan(oof[:, ci])
            yy, pp = y[m, ci].astype(int), oof[m, ci]
            row: dict[str, Any] = {"n": int(m.sum()), "pos": int(yy.sum()), "videos_pos": len({items[i]["video"] for i in np.where(m)[0] if y[i, ci] == 1}),
                                   "auc": roc_auc(yy, pp), "ap": average_precision(yy, pp), **calibration(yy, pp)}
            if v in scene:
                for fld in compare.get(c, ()):
                    vals = [(int(y[i, ci]), scene[v].get((items[i]["video"], items[i]["t_ms"]), {}).get(fld))
                            for i in np.where(m)[0]]
                    vals = [(a, b) for a, b in vals if b is not None]
                    if vals:
                        ya, pa = np.array([a for a, _ in vals]), np.array([b for _, b in vals], dtype=float)
                        row[f"auc_{fld}"] = roc_auc(ya, pa)
                        row[f"ap_{fld}"] = average_precision(ya, pa)
            summary[c][v] = row

    def fmt(val: Optional[float]) -> str:
        return "  —  " if val is None else f"{val:.3f}"

    print(f"\nOOF по фолдам видео (обучение: {', '.join(args.train_variants)}; l2 {args.l2:g}). AUC / AP; "
          "для orig и cache_* — AUC полей сцены на тех же кадрах:")
    for c in classes:
        print(f"[{c}]  {'вариант':<12}{'n':>6}{'pos':>6}{'вид+':>5}{'AUC':>7}{'AP':>7}   pos q05/25/50        neg q50/95/99/max      FP@.5  сцена")
        for v, r in summary[c].items():
            sc = "  ".join(f"{k_[4:]} {fmt(val)}" for k_, val in r.items() if k_.startswith("auc_"))
            print(f"      {v:<12}{r['n']:>6}{r['pos']:>6}{r['videos_pos']:>5}{fmt(r['auc']):>7}{fmt(r['ap']):>7}   "
                  f"{str(r['pos_q05_25_50']):<20} {str(r['neg_q50_95_99_max']):<24} {r['fp@0.5']:>4}  {sc}")

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"head": args.head, "zone": spec["zone"], "classes": list(classes), "l2": args.l2,
            "train_variants": args.train_variants, "folds_from": str(folds_path) if folds_path else None,
            "n": {c: int(((y[:, i] >= 0) & train_var).sum()) for i, c in enumerate(classes)},
            "pos": {c: int(((y[:, i] == 1) & train_var).sum()) for i, c in enumerate(classes)},
            "embedding": "google/siglip-base-patch16-224 get_image_features, L2-нормированный",
            "cv": {c: {v: summary[c][v]["auc"] for v in summary[c]} for c in classes}}
    np.savez(args.out / f"{args.head}.npz", w=final_w, b=final_b, classes=np.array(classes),
             meta=json.dumps(meta, ensure_ascii=False))
    args.report.mkdir(parents=True, exist_ok=True)
    np.savez(args.report / f"{args.head}_folds.npz", w=fold_w, b=fold_b, classes=np.array(classes),
             groups=json.dumps(folds, ensure_ascii=False))
    np.savez_compressed(args.report / f"{args.head}_oof.npz", variant=var, video=np.array([it["video"] for it in items]),
                        t=np.array([it["t"] for it in items], dtype=np.float32),
                        t_ms=np.array([it["t_ms"] for it in items], dtype=np.int64),
                        y=y, p=oof, fold=fold.astype(np.int8), classes=np.array(classes))
    (args.report / f"{args.head}_report.json").write_text(
        json.dumps({"summary": summary, "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nголова -> {args.out / (args.head + '.npz')} (w {final_w.shape}, b {final_b.shape}, классы {list(classes)})")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Обучение головы на эмбеддингах SigLIP.")
    p.add_argument("head", choices=sorted(HEADS))
    p.add_argument("--emb", type=Path, default=ROOT / "data" / "emb")
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw_cache_r3", help="для AUC zero-shot")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--train-variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--out", type=Path, default=ROOT / "models" / "heads")
    p.add_argument("--report", type=Path, default=ROOT / "data" / "ml_results" / "r3" / "heads")
    p.add_argument("--folds-from", type=Path, default=None,
                   help="готовое разбиение группа -> фолд (*_folds.npz или JSON); для desk_items по умолчанию — "
                        "фолды safe_open (data/ml_results/r3/heads/safe_open_folds.npz), чтобы головы и "
                        "многозадачное дообучение сравнивались на одних фолдах")
    p.add_argument("--classes", nargs="+", default=None,
                   help="desk_items: какие классы учить (по умолчанию train_classes: cash bag)")
    p.add_argument("--aug-emb", type=Path, default=ROOT / "data" / "emb_aug",
                   help="desk_items: эмбеддинги кадров аугментированных кэшей <v>/<zone>/orig — варианты cache_<v>")
    p.add_argument("--scene", type=Path, default=ROOT / "data" / "scene_cache_r3",
                   help="desk_items: сцены для сравнения с OWLv2/zero-shot (cache_<v> — <scene>/../scene_cache_aug_<v>)")
    args = p.parse_args(argv)

    spec = HEADS[args.head]
    aliases = {normalize_video_id(k): v for k, v in known_aliases().items()}
    if "classes" in spec:
        return train_multilabel(args, spec, aliases)
    labels = load_labels(args.labels or ROOT / "training" / "labels" / f"{args.head}.csv")
    items = load_data(args.head, args.emb, labels, aliases, args.raw)
    if not items:
        print("нет данных: сначала scripts/embed_zones.py embed и метки", file=sys.stderr)
        return 2
    folds = load_folds(args.folds_from) if args.folds_from else assign_folds(items, args.folds)
    x = np.stack([it["emb"] for it in items])
    y = np.array([it["y"] for it in items])
    var = np.array([it["variant"] for it in items])
    fold = np.array([folds[it["group"]] for it in items])
    labeled = y >= 0
    train_mask_var = np.isin(var, args.train_variants) & labeled

    counts = {v: {"n": int(((var == v) & labeled).sum()), "pos": int((y[var == v] == 1).sum())} for v in VARIANTS}
    print(f"[{args.head}] вырезок: " + ", ".join(f"{v} {c['n']} (open {c['pos']})" for v, c in counts.items()))

    oof = np.full(len(items), np.nan)
    per_fold = []
    fold_w = np.zeros((args.folds, x.shape[1]), dtype=np.float32)
    fold_b = np.zeros(args.folds, dtype=np.float32)
    for f in range(args.folds):
        tr = (fold != f) & train_mask_var
        te = fold == f
        if y[tr].sum() == 0:
            continue
        w, b = fit_logreg(x[tr], y[tr], args.l2)
        fold_w[f], fold_b[f] = w, b
        oof[te] = predict(x[te], w, b)
        te = te & labeled
        row: dict[str, Any] = {"fold": f, "videos": sorted({items[i]["group"] for i in np.where(te & (var == "orig"))[0]})}
        for v in VARIANTS:
            m = te & (var == v)
            row[v] = {"auc": roc_auc(y[m], oof[m]), "acc": float(((oof[m] >= 0.5) == y[m]).mean()) if m.any() else None,
                      "pos": int(y[m].sum()), "n": int(m.sum())}
        per_fold.append(row)

    zs = zero_shot_lookup(args.raw, spec["zero_shot"]) if spec["zero_shot"] else {}
    summary: dict[str, Any] = {}
    for v in VARIANTS:
        m = (var == v) & labeled
        if not m.any():
            continue
        yy, pp = y[m], oof[m]
        tp = int(((pp >= 0.5) & (yy == 1)).sum())
        fp = int(((pp >= 0.5) & (yy == 0)).sum())
        summary[v] = {"auc": roc_auc(yy, pp), "acc": float(((pp >= 0.5) == yy).mean()),
                      "recall_open": tp / max(int(yy.sum()), 1), "false_open": fp, "closed": int((yy == 0).sum())}
    zs_y, zs_p = [], []
    for it in items:
        if it["variant"] == "orig" and it["y"] >= 0:
            val = zs.get(it["video"], {}).get(int(round(it["t"] * 1000)))
            if val is not None:
                zs_y.append(it["y"])
                zs_p.append(val)
    summary["zero_shot_orig"] = {"auc": roc_auc(np.array(zs_y), np.array(zs_p)), "n": len(zs_y)}

    print("\nAUC вне фолда (разбиение по видео):")
    print(f"{'вариант':<10}{'AUC':>7}{'точн.':>8}{'полнота open':>14}{'ложн. open':>12}")
    for v in VARIANTS:
        s = summary.get(v)
        if s:
            auc = f"{s['auc']:.3f}" if s["auc"] is not None else "—"
            print(f"{v:<10}{auc:>7}{s['acc']:>8.3f}{s['recall_open']:>14.3f}{s['false_open']:>7} из {s['closed']}")
    zs_auc = summary["zero_shot_orig"]["auc"]
    print(f"zero-shot SigLIP на оригинале: AUC {zs_auc if zs_auc is None else round(zs_auc, 3)} (n={summary['zero_shot_orig']['n']})")
    print("\nпо фолдам (оригинал):")
    for r in per_fold:
        o = r["orig"]
        auc = f"{o['auc']:.3f}" if o["auc"] is not None else "  —  "
        print(f"  фолд {r['fold']}: AUC {auc}  open {o['pos']}/{o['n']}  видео {', '.join(r['videos'])}")

    # Финальная голова — на всех данных выбранных вариантов.
    w, b = fit_logreg(x[train_mask_var], y[train_mask_var], args.l2)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"head": args.head, "zone": spec["zone"], "l2": args.l2, "train_variants": args.train_variants,
            "n": int(train_mask_var.sum()), "pos": int(y[train_mask_var].sum()),
            "embedding": "google/siglip-base-patch16-224 get_image_features, L2-нормированный",
            "cv": {v: summary[v]["auc"] for v in VARIANTS if v in summary}}
    np.savez(args.out / f"{args.head}.npz", w=w, b=np.float32(b), meta=json.dumps(meta, ensure_ascii=False))
    args.report.mkdir(parents=True, exist_ok=True)
    oof_out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it, pr in zip(items, oof):
        oof_out[it["variant"]][it["video"]].append([it["t"], round(float(pr), 4), it["y"], *([it["track"]] if "track" in it else [])])
    np.savez(args.report / f"{args.head}_folds.npz", w=fold_w, b=fold_b,
             groups=json.dumps(folds, ensure_ascii=False))
    (args.report / f"{args.head}_oof.json").write_text(json.dumps(oof_out, ensure_ascii=False), encoding="utf-8")
    (args.report / f"{args.head}_report.json").write_text(
        json.dumps({"summary": summary, "folds": per_fold, "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nголова -> {args.out / (args.head + '.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
