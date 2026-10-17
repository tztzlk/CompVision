#!/usr/bin/env python3
"""Копии видео датасета «под другим ракурсом и светом» — проверка устойчивости и аугментация голов.

    PYTHONPATH=. .venv/bin/python scripts/augment_videos.py                       # все варианты
    PYTHONPATH=. .venv/bin/python scripts/augment_videos.py --variant flip --only Х-20

Для каждого <dataset>/*.mov пишется <out>/<вариант>/<то же имя>.mov — разметка подходит как есть
(eval_cache.py опознаёт видео по имени файла). Варианты:
  flip  — отражение по горизонтали (дверь и сейф меняются сторонами);
  persp — перспектива: углы исходного кадра сдвинуты внутрь на 6–10 % и растянуты на весь кадр
          (другая точка камеры без чёрных полей);
  crop  — окно 85 % со смещением к краю, растянутое обратно;
  light — яркость, контраст и баланс цвета.
Параметры фиксированные (seed): один набор на вариант для всех видео, не случайные на каждый кадр.
Кодек — H.264 (libx264, yuv420p), fps и размер кадра — как у оригинала, звук не копируется.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("flip", "persp", "crop", "light")
SEED = 26


def variant_params(variant: str, seed: int = SEED) -> dict[str, Any]:
    """Один набор параметров на вариант (доли кадра, множители)."""
    rng = np.random.default_rng([seed, VARIANTS.index(variant)])
    if variant == "flip":
        return {}
    if variant == "persp":
        # (dx, dy) сдвиг внутрь для углов TL, TR, BR, BL. dx у всех 6–10 %; dy 6–10 % только у одной
        # боковой стороны (у другой 0–2 %) — кадр сжимается к одному краю, как у камеры, повёрнутой вбок.
        shifts = rng.uniform(0.06, 0.10, size=(4, 2))
        weak = (1, 2) if rng.random() < 0.5 else (0, 3)
        for i in weak:
            shifts[i, 1] = rng.uniform(0.0, 0.02)
        return {"shifts": shifts.round(4).tolist()}
    if variant == "crop":
        return {"scale": 0.85, "anchor": [float(rng.choice([0.0, 1.0])), float(rng.uniform(0.0, 0.3))]}
    if variant == "light":
        return {
            "brightness": float(rng.choice([-1, 1]) * rng.uniform(0.18, 0.25)),
            "contrast": float(rng.choice([-1, 1]) * rng.uniform(0.10, 0.15)),
            "bgr_gain": rng.uniform(0.94, 1.06, size=3).round(4).tolist(),
        }
    raise ValueError(f"неизвестный вариант {variant!r}")


def make_transform(variant: str, params: dict[str, Any], w: int, h: int) -> Callable[[np.ndarray], np.ndarray]:
    if variant == "flip":
        return lambda f: cv2.flip(f, 1)
    if variant == "persp":
        s = np.asarray(params["shifts"], dtype=np.float32)
        src = np.float32([
            [s[0, 0] * w, s[0, 1] * h],
            [w - 1 - s[1, 0] * w, s[1, 1] * h],
            [w - 1 - s[2, 0] * w, h - 1 - s[2, 1] * h],
            [s[3, 0] * w, h - 1 - s[3, 1] * h],
        ])
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        m = cv2.getPerspectiveTransform(src, dst)
        return lambda f: cv2.warpPerspective(f, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if variant == "crop":
        cw, ch = int(round(w * params["scale"])), int(round(h * params["scale"]))
        x0 = int(round((w - cw) * params["anchor"][0]))
        y0 = int(round((h - ch) * params["anchor"][1]))
        return lambda f: cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    if variant == "light":
        gain = np.asarray(params["bgr_gain"], dtype=np.float32) * (1.0 + params["brightness"])
        c = 1.0 + params["contrast"]

        def light(f: np.ndarray) -> np.ndarray:
            x = f.astype(np.float32)
            x = (x - 128.0) * c + 128.0
            x = x * gain
            return np.clip(x, 0, 255).astype(np.uint8)
        return light
    raise ValueError(f"неизвестный вариант {variant!r}")


def transform_bbox(variant: str, params: dict[str, Any], bbox: Any, w: int, h: int) -> tuple[float, float, float, float]:
    """Рамка (x1, y1, x2, y2 в долях кадра) после преобразования варианта — охватывающая, в пределах кадра."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if variant == "light":
        return (x1, y1, x2, y2)
    if variant == "flip":
        return (1.0 - x2, y1, 1.0 - x1, y2)
    pts = np.float32([[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]])
    if variant == "crop":
        cw, ch = w * params["scale"], h * params["scale"]
        x0, y0 = round((w - round(cw)) * params["anchor"][0]), round((h - round(ch)) * params["anchor"][1])
        out = (pts - np.float32([x0, y0])) * np.float32([w / round(cw), h / round(ch)])
    elif variant == "persp":
        s = np.asarray(params["shifts"], dtype=np.float32)
        src = np.float32([[s[0, 0] * w, s[0, 1] * h], [w - 1 - s[1, 0] * w, s[1, 1] * h],
                          [w - 1 - s[2, 0] * w, h - 1 - s[2, 1] * h], [s[3, 0] * w, h - 1 - s[3, 1] * h]])
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        out = cv2.perspectiveTransform(pts[None], cv2.getPerspectiveTransform(src, dst))[0]
    else:
        raise ValueError(f"неизвестный вариант {variant!r}")
    xs, ys = np.clip(out[:, 0] / w, 0, 1), np.clip(out[:, 1] / h, 0, 1)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def augment_one(src: Path, dst: Path, variant: str, params: dict[str, Any]) -> tuple[str, int, float]:
    t0 = time.time()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"не открылось {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fn = make_transform(variant, params, w, h)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mov")
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-threads", "2", str(tmp)],
        stdin=subprocess.PIPE,
    )
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            proc.stdin.write(np.ascontiguousarray(fn(frame)).tobytes())
            n += 1
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0 or n == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg rc={rc}, кадров {n}: {src}")
    tmp.replace(dst)
    return src.stem, n, time.time() - t0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Аугментированные копии видео датасета.")
    p.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "videos")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "aug")
    p.add_argument("--variant", action="append", choices=VARIANTS, help="по умолчанию — все")
    p.add_argument("--only", action="append", help="video_id (имя файла без .mov), можно несколько")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true", help="перезаписать готовые файлы")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    videos = sorted(p for p in args.dataset.glob("*.mov"))
    if args.only:
        videos = [p for p in videos if p.stem in set(args.only)]
    if not videos:
        print(f"видео не найдены в {args.dataset}", file=sys.stderr)
        return 2
    errors = 0
    for variant in args.variant or VARIANTS:
        params = variant_params(variant)
        out_dir = args.out / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_params.json").write_text(
            json.dumps({"variant": variant, "seed": SEED, "params": params}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        todo = [v for v in videos if args.force or not (out_dir / v.name).exists()]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(augment_one, v, out_dir / v.name, variant, params) for v in todo]
            for f in futs:
                try:
                    f.result()
                except Exception as exc:          # одно битое видео не останавливает остальные
                    errors += 1
                    print(f"[{variant}] ошибка: {exc}", file=sys.stderr)
        print(f"[{variant}] {len(todo)} видео за {time.time() - t0:.0f} с -> {out_dir}  params={params}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
