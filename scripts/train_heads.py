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


def label_at(segments: list[tuple[float, float, str]], t: float) -> Optional[str]:
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
        out[normalize_video_id(raw["video_id"])] = {round(float(t), 1): p for t, p in raw["state_samples"].get(key, [])
                                                    if p is not None}
    return out


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
    args = p.parse_args(argv)

    spec = HEADS[args.head]
    labels = load_labels(args.labels or ROOT / "training" / "labels" / f"{args.head}.csv")
    aliases = {normalize_video_id(k): v for k, v in known_aliases().items()}
    items = load_data(args.head, args.emb, labels, aliases, args.raw)
    if not items:
        print("нет данных: сначала scripts/embed_zones.py embed и метки", file=sys.stderr)
        return 2
    folds = assign_folds(items, args.folds)
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
            val = zs.get(it["video"], {}).get(round(it["t"], 1))
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
