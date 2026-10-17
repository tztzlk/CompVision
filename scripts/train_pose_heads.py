#!/usr/bin/env python3
"""Обучение голов жестов кассира поверх точек позы — обучаемая часть гибрида (worker/scene/pose_heads.py).

    PYTHONPATH=. python scripts/train_pose_heads.py
    PYTHONPATH=. python scripts/train_pose_heads.py --variants orig flip persp --folds 5

Данные: сцены с позой (scripts/add_pose_to_cache.py) — оригинал data/scene_cache_r3_pose и аугментации
data/scene_cache_aug_<вариант>_pose; метки — training/labels/pose_gestures.csv: по плитке на секунду
(scripts/pose_label_sheet.py), два независимых разметчика, расхождения решает третий. Метка состояния —
метка ближайшей плитки не дальше 0,5 с; unsure и плитки без кассира не учим. Аугментации берут метки
оригинала (то же время, тот же ролик).

Проверка — только по видео: K фолдов по группам видео, все варианты видео — в фолде оригинала.
Дубликаты Х-47≡Х-4 и Х-48≡Х-5 в обучение не идут (метки размечены по Х-4 и Х-5). Соседние кадры
одного ролика почти одинаковы: разбиение по кадрам дало бы фальшивые 99 %.

Модели на жест: логистическая регрессия (L2) и MLP с одним скрытым слоем (torch, только здесь;
в worker — numpy). Выбор модели — по ROC-AUC вне фолда на оригиналах; порог — по F1 вне фолда (сетка
0,05…0,95: с весами классов вероятности редких жестов смещены вверх), вшивается в смещение (0,5 = порог).
Режим гибрида (geometry / band / fill / head) — по F1 при пороге 0,5 вне фолда: среднее «оригинал» и
«среднее по аугментациям»; голова включается, только если обгоняет геометрию не меньше чем на --min-gain.
Веса и стандартизация — честно вне фолда; выбор модели, порога и режима — по общим вероятностям вне
фолда всех видео (без вложенной проверки), поэтому метрики — оценка сверху. Всё меряется на тех же
40 видео одной комнаты: другой камеры в данных нет.
Геометрия (входы geo_* и база сравнения) пересчитывается из точек сцены, а не берётся из полей кэша:
кэш, прошедший add_pose_to_cache --gesture-heads, хранит в этих полях уже гибрид.

Выход: models/heads/pose_gestures.npz (головы на всех данных), data/ml_results/pose/heads_report.json,
data/ml_results/pose/heads_oof.csv (вероятности вне фолда) и таблица в stdout. С --oof-scenes DIR — ещё и
сцены всех вариантов с гибридом голов ВНЕ фолда (DIR/<вариант>/<видео>.json: каждое видео — головами,
обученными без него) — для честного замера детекторов позы (scripts/eval_pose_detectors.py).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from worker.scene.contract import load_scene, save_scene  # noqa: E402
from worker.scene.pose_features import compute_features  # noqa: E402
from worker.scene.pose_heads import (  # noqa: E402
    FEATURE_NAMES, GEOMETRY_OF, GESTURES, MODES, GestureHead, apply_gesture_heads, hybrid_value, window_features,
)
from worker.scene.pose_pass import pose_frames_from_scene  # noqa: E402

VARIANT_DIRS = {
    "orig": "data/scene_cache_r3_pose",
    "flip": "data/scene_cache_aug_flip_pose",
    "persp": "data/scene_cache_aug_persp_pose",
    "crop": "data/scene_cache_aug_crop_pose",
    "light": "data/scene_cache_aug_light_pose",
}
DUPLICATES = {"Х-47": "Х-4", "Х-48": "Х-5"}
YES_NO = {"yes": 1, "no": 0}
BAND = (0.2, 0.8)
SEED = 26


# --------------------------------------------------------------------------- данные


def norm_id(video_id: str) -> str:
    """Ключ видео без различия латинской и кириллической «Х» (X-1 и Х-3 в разметке датасета)."""
    return video_id.replace("X", "Х").replace("x", "х")


def load_labels(path: Path) -> dict[str, dict[int, dict[str, Optional[int]]]]:
    """video_id -> t (с) -> жест -> 1 / 0 / None (unsure или кассира нет)."""
    out: dict[str, dict[int, dict[str, Optional[int]]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            visible = row["cashier_visible"].strip() == "yes"
            out[norm_id(row["video_id"].strip())][int(row["t"])] = {
                g: (YES_NO.get(row[g].strip()) if visible else None) for g in GESTURES}
    return out


def build_samples(variants: list[str], labels: dict[str, dict[int, dict[str, Optional[int]]]]) -> list[dict[str, Any]]:
    """Состояния с точками кассира и меткой ближайшей плитки: признаки, геометрия, метки."""
    samples: list[dict[str, Any]] = []
    for variant in variants:
        cache = ROOT / VARIANT_DIRS[variant]
        for path in sorted(cache.glob("*.json")):
            if path.stem in DUPLICATES or path.name.startswith("_"):
                continue
            tiles = labels.get(norm_id(path.stem))
            if not tiles:
                continue
            scene = load_scene(path)
            frames = pose_frames_from_scene(scene)
            geometry = compute_features(frames, scene.zones, scene.frame_size)
            X = window_features(frames, geometry, scene.zones, scene.frame_size)
            for st, x, geo in zip(scene.states, X, geometry):
                if x is None:
                    continue
                t = int(round(st.t))
                if abs(st.t - t) > 0.5 + 1e-6 or t not in tiles:
                    continue
                samples.append({"variant": variant, "video": norm_id(path.stem), "video_id": path.stem,
                                "t": st.t, "x": x, "geo": geo, "y": tiles[t]})
    return samples


def make_folds(videos: list[str], k: int) -> dict[str, int]:
    order = sorted(videos)
    random.Random(SEED).shuffle(order)
    return {v: i % k for i, v in enumerate(order)}


# --------------------------------------------------------------------------- метрики


def roc_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    """ROC-AUC через ранги (Манна — Уитни), связи — средним рангом. None — один класс."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def prf(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": round(f1, 3), "precision": round(precision, 3), "recall": round(recall, 3), "tp": tp, "fp": fp, "fn": fn}


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 3)


# --------------------------------------------------------------------------- модели (torch)


def fit(X: np.ndarray, y: np.ndarray, kind: str, l2: float, hidden: int, epochs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Обучить голову на стандартизованных X; веса классов — обратно частоте. Возвращает слои [(W, b)]."""
    import torch

    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    pos = float(y.sum())
    pos_weight = torch.tensor([(len(y) - pos) / pos if pos > 0 else 1.0])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if kind == "lr":
        model = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def closure() -> Any:
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt) + l2 * (model.weight ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return [(model.weight.detach().numpy().T.astype(np.float64), model.bias.detach().numpy().astype(np.float64))]
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=l2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    return [(model[0].weight.detach().numpy().T.astype(np.float64), model[0].bias.detach().numpy().astype(np.float64)),
            (model[2].weight.detach().numpy().T.astype(np.float64), model[2].bias.detach().numpy().astype(np.float64))]


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def head_of(gesture: str, mean: np.ndarray, std: np.ndarray, layers: list, mode: str = "head") -> GestureHead:
    return GestureHead(gesture=gesture, mean=mean, std=std, layers=tuple(layers), mode=mode,
                       band_lo=BAND[0], band_hi=BAND[1])


def shift_threshold(layers: list, threshold: float) -> list:
    """Вшить порог в смещение последнего слоя: вероятность 0,5 после сдвига = прежний порог."""
    W, b = layers[-1]
    logit = math.log(threshold / (1 - threshold))
    return layers[:-1] + [(W, b - logit)]


# --------------------------------------------------------------------------- обучение и замер


def shift_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Те же вероятности после вшитого порога: threshold -> 0,5 (NaN остаются NaN)."""
    p = np.clip(probs, 1e-9, 1 - 1e-9)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - math.log(threshold / (1 - threshold)))))


def evaluate(samples: list[dict[str, Any]], gesture: str, probs: np.ndarray, variants: list[str]) -> dict[str, Any]:
    """Метрики вне фолда по вариантам: AUC головы и геометрии, F1 геометрии, головы и режимов гибрида."""
    field = GEOMETRY_OF[gesture]
    idx = [i for i, s in enumerate(samples) if s["y"][gesture] is not None]
    rows = [(samples[i], p) for i, p in zip(idx, probs) if not np.isnan(p)]
    out: dict[str, Any] = {}
    for variant in variants:
        part = [(s, p) for s, p in rows if s["variant"] == variant]
        if not part:
            continue
        y = np.asarray([s["y"][gesture] for s, _ in part])
        p = np.asarray([p for _, p in part])
        geo = [s["geo"].get(field) for s, _ in part]
        g = np.asarray([v if v is not None else 0.0 for v in geo])
        entry: dict[str, Any] = {
            "n": len(part), "pos": int(y.sum()), "geo_none": sum(1 for v in geo if v is None),
            "auc_head": _r(roc_auc(y, p)), "auc_geometry": _r(roc_auc(y, g)),
        }
        for mode in MODES:
            head = GestureHead(gesture, None, None, (), mode, BAND[0], BAND[1])
            values = [hybrid_value(v, float(q), head) for v, q in zip(geo, p)]
            pred = np.asarray([1 if (v is not None and v >= 0.5) else 0 for v in values])
            entry[mode] = prf(y, pred)
        out[variant] = entry
    return out


def mode_score(metrics: dict[str, Any], mode: str) -> float:
    orig = metrics.get("orig", {}).get(mode, {}).get("f1", 0.0)
    aug = [m[mode]["f1"] for v, m in metrics.items() if v != "orig"]
    return 0.5 * orig + 0.5 * (sum(aug) / len(aug) if aug else orig)


def best_threshold(samples: list[dict[str, Any]], gesture: str, probs: np.ndarray) -> float:
    """Порог головы по F1 вне фолда на оригиналах (сетка 0,05…0,95)."""
    idx = [i for i, s in enumerate(samples) if s["y"][gesture] is not None]
    pairs = [(samples[i]["y"][gesture], p) for i, p in zip(idx, probs)
             if samples[i]["variant"] == "orig" and not np.isnan(p)]
    if not pairs:
        return 0.5
    y = np.asarray([a for a, _ in pairs])
    p = np.asarray([b for _, b in pairs])
    grid = [round(0.05 + 0.05 * i, 2) for i in range(19)]
    return max(grid, key=lambda thr: (prf(y, (p >= thr).astype(int))["f1"], -abs(thr - 0.5)))


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Головы жестов кассира по точкам позы (проверка по видео).")
    p.add_argument("--labels", type=Path, default=ROOT / "training" / "labels" / "pose_gestures.csv")
    p.add_argument("--variants", nargs="+", default=list(VARIANT_DIRS), choices=list(VARIANT_DIRS))
    p.add_argument("--train-variants", nargs="+", default=None, choices=list(VARIANT_DIRS),
                   help="на чём учить (по умолчанию — все --variants); мерить — на всех --variants")
    p.add_argument("--gestures", nargs="+", default=list(GESTURES), choices=list(GESTURES))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--min-gain", type=float, default=0.02, help="насколько голова должна обогнать геометрию по F1")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "heads" / "pose_gestures.npz")
    p.add_argument("--report", type=Path, default=ROOT / "data" / "ml_results" / "pose")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--oof-scenes", type=Path, default=None,
                   help="записать сцены с гибридом голов вне фолда: DIR/<вариант>/<видео>.json")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    torch.set_num_threads(max(1, args.threads))
    started = time.monotonic()
    labels = load_labels(args.labels)
    samples = build_samples(args.variants, labels)
    train_variants = set(args.train_variants or args.variants)
    videos = sorted({s["video"] for s in samples})
    folds = make_folds(videos, args.folds)
    per_variant = ", ".join(f"{v} {sum(1 for s in samples if s['variant'] == v)}" for v in args.variants)
    print(f"меток: {sum(len(v) for v in labels.values())} плиток по {len(labels)} видео; состояний с меткой: "
          f"{len(samples)} ({per_variant}); признаков {len(FEATURE_NAMES)}; фолдов {args.folds}")

    candidates = [("lr", 1e-3), ("lr", 1e-2), ("mlp", 1e-3)]
    report: dict[str, Any] = {"created_by": "scripts/train_pose_heads.py", "git": git_head(), "folds": args.folds,
                              "variants": args.variants, "train_variants": sorted(train_variants),
                              "n_features": len(FEATURE_NAMES), "heads": {}}
    npz: dict[str, Any] = {"feature_names": np.asarray(FEATURE_NAMES)}
    meta_heads: dict[str, Any] = {}
    oof_rows: list[list[Any]] = []
    fold_models: dict[str, tuple[dict[int, Any], float, str]] = {}
    for gesture in args.gestures:
        # Обучение — только на train_variants; замер вне фолда — на всех вариантах.
        train_mask = [s["variant"] in train_variants for s in samples]
        scored: list[tuple[float, str, float, np.ndarray, dict]] = []
        for kind, l2 in candidates:
            probs, models = oof_with_mask(samples, gesture, folds, args.folds, kind, l2, args.hidden, args.epochs,
                                          train_mask)
            idx = [i for i, s in enumerate(samples) if s["y"][gesture] is not None]
            y = np.asarray([samples[i]["y"][gesture] for i in idx])
            orig = np.asarray([samples[i]["variant"] == "orig" for i in idx])
            ok = orig & ~np.isnan(probs)
            auc = roc_auc(y[ok], probs[ok]) if ok.any() else None
            scored.append((auc if auc is not None else 0.0, kind, l2, probs, models))
            print(f"  {gesture}: {kind} l2={l2:g} AUC вне фолда (оригиналы) {_r(auc)}")
        auc, kind, l2, probs, models = max(scored, key=lambda item: item[0])
        threshold = best_threshold(samples, gesture, probs)
        shifted = shift_probs(probs, threshold)
        metrics = evaluate(samples, gesture, shifted, args.variants)
        scores = {mode: round(mode_score(metrics, mode), 3) for mode in MODES}
        best_mode = max((m for m in MODES if m != "geometry"), key=lambda m: scores[m])
        mode = best_mode if scores[best_mode] >= scores["geometry"] + args.min_gain else "geometry"
        fold_models[gesture] = (models, threshold, mode)

        idx = [i for i, s in enumerate(samples) if s["y"][gesture] is not None and train_mask[i]]
        X = np.asarray([samples[i]["x"] for i in idx], dtype=np.float64)
        y = np.asarray([samples[i]["y"][gesture] for i in idx], dtype=np.float64)
        mean, std = standardize(X)
        layers = shift_threshold(fit((X - mean) / std, y, kind, l2, args.hidden, args.epochs), threshold)
        npz[f"{gesture}__mean"], npz[f"{gesture}__std"] = mean, std
        for k, (W, b) in enumerate(layers):
            npz[f"{gesture}__W{k}"], npz[f"{gesture}__b{k}"] = W, b
        meta_heads[gesture] = {"mode": mode, "layers": len(layers), "kind": kind, "l2": l2, "threshold": threshold,
                               "band_lo": BAND[0], "band_hi": BAND[1], "auc_oof_orig": _r(auc),
                               "train_states": len(idx), "train_pos": int(y.sum())}
        report["heads"][gesture] = {**meta_heads[gesture], "mode_scores": scores, "metrics": metrics,
                                    "videos_with_pos": sorted({samples[i]["video_id"] for i in idx if samples[i]["y"][gesture] == 1}),
                                    "threshold_on_grid_edge": threshold in (0.05, 0.95)}
        lab = [i for i, s in enumerate(samples) if s["y"][gesture] is not None]
        for i, p in zip(lab, shifted):
            s = samples[i]
            oof_rows.append([gesture, s["variant"], s["video_id"], round(s["t"], 3), s["y"][gesture],
                             _r(s["geo"].get(GEOMETRY_OF[gesture])), None if np.isnan(p) else round(float(p), 3)])
        line = " ".join(f"{m} {scores[m]:.3f}" for m in MODES)
        print(f"{gesture}: модель {kind} (l2 {l2:g}), порог {threshold}, режим -> {mode}; F1-оценка режимов: {line}")

    meta = {"version": 1, "git": report["git"], "gestures": list(args.gestures), "heads": meta_heads,
            "labels": str(args.labels.relative_to(ROOT)) if args.labels.is_relative_to(ROOT) else str(args.labels),
            "train_variants": sorted(train_variants)}
    npz["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **npz)
    with (args.report / "heads_oof.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gesture", "variant", "video_id", "t", "label", "geometry", "head_oof"])
        w.writerows(oof_rows)
    if args.oof_scenes:
        written, headed = write_oof_scenes(args.oof_scenes, args.variants, folds, fold_models)
        report["oof_scenes"] = {"written": written, "with_heads": headed}
        print(f"сцены вне фолда: {written} -> {args.oof_scenes}; с головой по жестам: {headed}")
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "heads_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print_table(report, args.variants)
    print(f"головы: {args.out}; отчёт: {args.report / 'heads_report.json'}; {time.monotonic() - started:.0f} с")
    return 0


def oof_with_mask(samples: list[dict[str, Any]], gesture: str, folds: dict[str, int], k: int, kind: str, l2: float,
                  hidden: int, epochs: int, train_mask: list[bool]) -> tuple[np.ndarray, dict[int, Any]]:
    """Вероятности вне фолда для всех состояний с меткой жеста (порядок — как в samples с меткой) и модели
    фолдов {фолд: (mean, std, слои)}. Учим только на состояниях train_mask (варианты для обучения)."""
    idx = [i for i, s in enumerate(samples) if s["y"][gesture] is not None]
    X = np.asarray([samples[i]["x"] for i in idx], dtype=np.float64)
    y = np.asarray([samples[i]["y"][gesture] for i in idx], dtype=np.float64)
    fold_of = np.asarray([folds[samples[i]["video"]] for i in idx])
    trainable = np.asarray([train_mask[i] for i in idx])
    out = np.full(len(idx), np.nan)
    models: dict[int, Any] = {}
    for f in range(k):
        train, test = (fold_of != f) & trainable, fold_of == f
        if not test.any() or y[train].sum() == 0 or y[train].sum() == train.sum():
            continue
        mean, std = standardize(X[train])
        layers = fit((X[train] - mean) / std, y[train], kind, l2, hidden, epochs)
        models[f] = (mean, std, layers)
        out[test] = head_of(gesture, mean, std, layers).predict(X[test])
    return out, models


def write_oof_scenes(out_dir: Path, variants: list[str], folds: dict[str, int],
                     fold_models: dict[str, tuple[dict[int, Any], float, str]]) -> tuple[int, dict[str, int]]:
    """Сцены с гибридом голов вне фолда: видео обрабатывается головами фолда, в обучении которого его не было.

    Видео вне фолдов (без меток) и фолды, где модель не обучилась (нет позитивов), остаются на геометрии —
    сколько сцен получили голову, возвращается по жестам."""
    written = 0
    headed: dict[str, int] = {g: 0 for g, (_, _, mode) in fold_models.items() if mode != "geometry"}
    for variant in variants:
        for path in sorted((ROOT / VARIANT_DIRS[variant]).glob("*.json")):
            if path.name.startswith("_"):
                continue
            fold = folds.get(norm_id(DUPLICATES.get(path.stem, path.stem)))
            heads: dict[str, GestureHead] = {}
            for gesture, (models, threshold, mode) in fold_models.items():
                if mode != "geometry" and fold in models:
                    mean, std, layers = models[fold]
                    heads[gesture] = head_of(gesture, mean, std, shift_threshold(layers, threshold), mode)
                    headed[gesture] += 1
            scene = load_scene(path)
            frames = pose_frames_from_scene(scene)
            features = compute_features(frames, scene.zones, scene.frame_size)
            rows = apply_gesture_heads(frames, features, scene.zones, scene.frame_size, heads)
            states = tuple(dataclasses.replace(st, extra={**{k: v for k, v in st.extra.items() if not k.endswith("_head")},
                                                          **row}) for st, row in zip(scene.states, rows))
            pose_meta = {**{k: v for k, v in scene.meta.get("pose", {}).items() if k not in ("gesture_heads", "heads_error")},
                         "gesture_heads_oof": {g: h.mode for g, h in sorted(heads.items())}}
            save_scene(dataclasses.replace(scene, states=states, meta={**scene.meta, "pose": pose_meta}),
                       out_dir / variant / path.name)
            written += 1
    return written, headed


def print_table(report: dict[str, Any], variants: list[str]) -> None:
    print()
    print("| жест | режим | вариант | n / поз. | AUC голова | AUC геометрия | F1 геометрия | F1 голова | F1 выбранный |")
    print("|---|---|---|---|---|---|---|---|---|")
    for gesture, info in report["heads"].items():
        for variant in variants:
            m = info["metrics"].get(variant)
            if not m:
                continue
            chosen = m[info["mode"]]["f1"]
            print(f"| {gesture} | {info['mode']} | {variant} | {m['n']} / {m['pos']} | {m['auc_head']} | "
                  f"{m['auc_geometry']} | {m['geometry']['f1']} | {m['head']['f1']} | {chosen} |")


if __name__ == "__main__":
    sys.exit(main())
