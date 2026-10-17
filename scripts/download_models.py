#!/usr/bin/env python3
"""Однократная загрузка весов моделей в каталог ``models/``.

Назначение
----------
По ТЗ сервис после скачивания весов обязан работать полностью офлайн
(облачные AI запрещены). Этот скрипт запускается ОДИН РАЗ на хосте с доступом
в интернет, скачивает все веса из декларативного списка ``MODELS`` в
``models/`` (относительно корня проекта) и пишет ``models/MANIFEST.json``
с фактическими sha256/размерами, чтобы контейнер/worker мог проверить
целостность весов уже без сети.

Зависимости: только стандартная библиотека Python (>= 3.9).

Примеры запуска
---------------
    python scripts/download_models.py                 # скачать всё, что ещё не скачано (~1,9 ГБ, из них ViTPose ~0,33 ГБ)
    python scripts/download_models.py --list          # показать список моделей и выйти
    python scripts/download_models.py --only rtdetr   # только группа (rtdetr, owlv2, siglip, vitpose) или файл
    python scripts/download_models.py --force         # перекачать даже уже существующие
    python scripts/download_models.py --models-dir /data/models
    MODELS_DIR=/data/models python scripts/download_models.py

Коды возврата: 0 — всё успешно (или список моделей пуст), 1 — хотя бы одна
модель не скачана / не прошла проверку sha256, 2 — ошибка аргументов.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

LOG = logging.getLogger("download_models")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_NAME = "MANIFEST.json"

USER_AGENT = "CompVision-model-downloader/1.0 (+python-urllib)"
CHUNK_SIZE = 1024 * 1024          # 1 МБ за одно чтение
PROGRESS_EVERY_MB = 10            # лог прогресса каждые N МБ
RETRIES = 3                       # попыток на сетевых ошибках
BACKOFF_BASE_SEC = 2.0            # пауза: 2, 4, 8 c
HTTP_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class ModelSpec:
    """Описание одной модели (одного файла весов).

    name    — короткое имя для логов и ``--only``.
    url     — прямая ссылка на файл (http/https).
    dest    — относительный путь внутри ``models/`` (например ``yolo/yolo11n.pt``).
    sha256  — ожидаемая контрольная сумма (hex). ``None`` — не проверять
              (не рекомендуется: тогда повторный запуск считает любой
              существующий файл валидным).
    size_mb — ориентировочный размер, только для лога/``--list``.
    group   — группа файлов одной модели (``--only rtdetr`` выбирает все её файлы).
    """

    name: str
    url: str
    dest: str
    sha256: Optional[str] = None
    size_mb: Optional[float] = None
    group: Optional[str] = None


# ---------------------------------------------------------------------------
# Список моделей восприятия (worker/perception/models.py, тот же набор и ревизии).
#
# Файлы репозиториев Hugging Face скачиваются по прямым ссылкам на зафиксированную ревизию
# (коммит) в models/hf/<каталог>/ — только stdlib, без huggingface_hub. Worker грузит их
# через from_pretrained(<каталог>, local_files_only=True), сеть в runtime не нужна.
# sha256 указан для больших файлов (из метаданных LFS на Hugging Face); мелкие конфиги
# неизменны благодаря ревизии в URL. Лицензии всех трёх моделей — Apache-2.0 (проверено
# на страницах моделей 17.10.2026). Ultralytics YOLO (AGPL-3.0) не используется.
# ---------------------------------------------------------------------------
_HF_URL = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _hf_files(group: str, repo: str, revision: str, local_dir: str,
              files: list[tuple[str, Optional[str], float]]) -> list[ModelSpec]:
    """Файлы одного репозитория Hugging Face (имя, sha256 | None, МБ) -> ModelSpec в models/hf/<local_dir>/."""
    return [
        ModelSpec(
            name=f"{group}/{filename}",
            url=_HF_URL.format(repo=repo, revision=revision, filename=filename),
            dest=f"hf/{local_dir}/{filename}",
            sha256=sha256,
            size_mb=size_mb,
            group=group,
        )
        for filename, sha256, size_mb in files
    ]


MODELS: list[ModelSpec] = [
    # RT-DETR R18 (COCO): люди и предметы на каждом анализируемом кадре.
    *_hf_files("rtdetr", "PekingU/rtdetr_r18vd_coco_o365", "386c43855d01bcd966e548f3296fbd2cd2a2a40b",
               "rtdetr_r18vd_coco_o365", [
                   ("config.json", None, 0.01),
                   ("preprocessor_config.json", None, 0.01),
                   ("model.safetensors", "eadade7a507b87cabe5d6a75b935dbd9fc59b1b843f36f7e3b24eafc856c0774", 80.9),
               ]),
    # OWLv2 base: зоны (стол, сейф, дверь, ноутбук) по текстовым запросам.
    *_hf_files("owlv2", "google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d",
               "owlv2-base-patch16-ensemble", [
                   ("config.json", None, 0.01),
                   ("preprocessor_config.json", None, 0.01),
                   ("tokenizer_config.json", None, 0.01),
                   ("special_tokens_map.json", None, 0.01),
                   ("added_tokens.json", None, 0.01),
                   ("vocab.json", None, 1.1),
                   ("merges.txt", None, 0.5),
                   ("model.safetensors", "e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7", 619.9),
               ]),
    # SigLIP base: zero-shot состояния по вырезкам зон (сейф открыт, деньги на столе...).
    *_hf_files("siglip", "google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
               "siglip-base-patch16-224", [
                   ("config.json", None, 0.01),
                   ("preprocessor_config.json", None, 0.01),
                   ("tokenizer_config.json", None, 0.01),
                   ("special_tokens_map.json", None, 0.01),
                   ("spiece.model", "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec", 0.8),
                   ("model.safetensors", "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8", 812.7),
               ]),
    # ViTPose base (simple decoder): 17 точек COCO людей — только при ENABLE_POSE=1
    # (worker/perception/pose.py). Apache-2.0 и ревизия проверены по API Hugging Face 17.10.2026.
    # Нужен transformers >= 4.48 (в worker/requirements.txt пока 4.46.3 — см. README, «Поза»).
    *_hf_files("vitpose", "usyd-community/vitpose-base-simple", "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
               "vitpose-base-simple", [
                   ("config.json", None, 0.01),
                   ("preprocessor_config.json", None, 0.01),
                   ("model.safetensors", "85375373893ddd3641f3912821073e53f5435f9e966e1dca59d004454bfe4fdf", 327.8),
               ]),
]


class DownloadError(Exception):
    """Ошибка скачивания или проверки модели."""


def sha256_of_file(path: Path) -> str:
    """Посчитать sha256 файла потоково (без загрузки в память)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} МБ"


def _download_once(url: str, part_path: Path) -> int:
    """Одна попытка скачать ``url`` в ``part_path``. Возвращает число байт."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    downloaded = 0
    next_report = PROGRESS_EVERY_MB * 1024 * 1024
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
        total_hdr = resp.headers.get("Content-Length")
        total = int(total_hdr) if total_hdr and total_hdr.isdigit() else None
        if total is not None:
            LOG.info("  размер по Content-Length: %s", _fmt_mb(total))
        with part_path.open("wb") as out:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        LOG.info("  %s / %s (%.0f%%)", _fmt_mb(downloaded),
                                 _fmt_mb(total), 100.0 * downloaded / total)
                    else:
                        LOG.info("  %s скачано", _fmt_mb(downloaded))
                    next_report += PROGRESS_EVERY_MB * 1024 * 1024
        if total is not None and downloaded != total:
            raise DownloadError(
                f"получено {downloaded} байт, ожидалось {total} (обрыв соединения?)")
    return downloaded


def download_with_retries(url: str, part_path: Path) -> int:
    """Скачать с повторами (``RETRIES`` попыток, экспоненциальный backoff)."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, RETRIES + 1):
        try:
            return _download_once(url, part_path)
        except urllib.error.HTTPError as exc:
            # 4xx — повторять бессмысленно (неверный URL / нет доступа).
            if 400 <= exc.code < 500:
                raise DownloadError(f"HTTP {exc.code} {exc.reason}: {url}") from exc
            last_exc = exc
        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last_exc = exc
        if attempt < RETRIES:
            delay = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            LOG.warning("  попытка %d/%d не удалась (%s), повтор через %.1f c",
                        attempt, RETRIES, last_exc, delay)
            time.sleep(delay)
    raise DownloadError(f"не удалось скачать после {RETRIES} попыток: {last_exc}") from last_exc


def process_model(spec: ModelSpec, models_dir: Path, force: bool = False) -> dict:
    """Скачать (или пропустить) одну модель.

    Возвращает запись для MANIFEST: name, dest, url, sha256 (фактический),
    size_bytes, status (``skipped`` | ``downloaded``), downloaded_at.
    Бросает ``DownloadError`` при любой неустранимой ошибке.
    """
    dest = models_dir / spec.dest
    part = dest.with_name(dest.name + ".part")
    expected = spec.sha256.lower() if spec.sha256 else None

    if dest.exists() and not force:
        if expected is None:
            LOG.info("[%s] уже есть: %s (sha256 не задан, проверка пропущена) -> skip",
                     spec.name, dest)
            return _manifest_entry(spec, dest, sha256_of_file(dest), "skipped")
        actual = sha256_of_file(dest)
        if actual == expected:
            LOG.info("[%s] уже есть и sha256 совпадает -> skip", spec.name)
            return _manifest_entry(spec, dest, actual, "skipped")
        LOG.warning("[%s] файл есть, но sha256 не совпадает (%s...), перекачиваю",
                    spec.name, actual[:12])

    dest.parent.mkdir(parents=True, exist_ok=True)
    if part.exists():
        part.unlink()

    size_hint = f", ~{spec.size_mb:.1f} МБ" if spec.size_mb else ""
    LOG.info("[%s] скачиваю %s -> %s%s", spec.name, spec.url, dest, size_hint)
    try:
        n_bytes = download_with_retries(spec.url, part)
        actual = sha256_of_file(part)
        if expected is not None and actual != expected:
            raise DownloadError(
                f"sha256 не совпадает: ожидалось {expected}, получено {actual}")
        # Атомарная замена: dest либо старый целиком, либо новый целиком.
        os.replace(part, dest)
    except BaseException:
        if part.exists():
            part.unlink()
        raise

    LOG.info("[%s] готово: %s (%s, sha256=%s...)", spec.name, dest, _fmt_mb(n_bytes), actual[:12])
    return _manifest_entry(spec, dest, actual, "downloaded")


def _manifest_entry(spec: ModelSpec, dest: Path, sha256: str, status: str) -> dict:
    return {
        "name": spec.name,
        "dest": spec.dest,
        "url": spec.url,
        "sha256": sha256,
        "sha256_expected": spec.sha256,
        "size_bytes": dest.stat().st_size,
        "status": status,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(models_dir: Path, entries: list[dict]) -> Path:
    """Записать ``models/MANIFEST.json`` атомарно (через временный файл)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / MANIFEST_NAME
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_dir": str(models_dir),
        "models": entries,
    }
    tmp = manifest_path.with_name(MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return manifest_path


def select_models(specs: Iterable[ModelSpec], only: Optional[list[str]]) -> list[ModelSpec]:
    """Отфильтровать по ``--only`` (группа модели или имя файла); неизвестные имена — ошибка."""
    specs = list(specs)
    if not only:
        return specs
    groups = sorted({s.group for s in specs if s.group})
    known = {s.name for s in specs} | set(groups)
    unknown = [n for n in only if n not in known]
    if unknown:
        raise SystemExit(f"Неизвестные модели в --only: {', '.join(unknown)}. "
                         f"Доступны группы: {', '.join(groups) or '(список пуст)'}")
    wanted = set(only)
    return [s for s in specs if s.name in wanted or (s.group is not None and s.group in wanted)]


def print_list(specs: list[ModelSpec], models_dir: Path) -> None:
    if not specs:
        print("Список моделей пока пуст — см. TODO у MODELS в scripts/download_models.py")
        return
    print(f"Каталог моделей: {models_dir}")
    for s in specs:
        present = "есть" if (models_dir / s.dest).exists() else "нет"
        size = f"~{s.size_mb:.1f} МБ" if s.size_mb else "размер не указан"
        sha = s.sha256[:12] + "..." if s.sha256 else "sha256 не задан"
        print(f"  {s.name:<20} {s.dest:<40} {size:<16} {sha:<18} [{present}]  {s.url}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Однократная загрузка весов моделей в models/ (офлайн-режим сервиса).")
    p.add_argument("--models-dir", type=Path,
                   default=Path(os.environ.get("MODELS_DIR") or DEFAULT_MODELS_DIR),
                   help="каталог для весов (по умолчанию $MODELS_DIR или <корень проекта>/models)")
    p.add_argument("--force", action="store_true", help="перекачать даже существующие файлы")
    p.add_argument("--list", action="store_true", help="показать список моделей и выйти")
    p.add_argument("--only", action="append", metavar="NAME",
                   help="скачать только указанную модель (флаг можно повторять)")
    p.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    models_dir: Path = args.models_dir.expanduser().resolve()

    if args.list:
        print_list(MODELS, models_dir)
        return 0

    if not MODELS:
        # Пустой список — не ошибка: проект на старте, веса ещё не выбраны.
        # MANIFEST.json не пишем и каталог не создаём, чтобы не оставлять
        # в репозитории пустой артефакт, который можно принять за валидный.
        LOG.info("Список моделей пока пуст — скачивать нечего. "
                 "См. TODO у MODELS в scripts/download_models.py")
        return 0

    specs = select_models(MODELS, args.only)
    LOG.info("Каталог моделей: %s (моделей к обработке: %d)", models_dir, len(specs))

    entries: list[dict] = []
    failed: list[str] = []
    for spec in specs:
        try:
            entries.append(process_model(spec, models_dir, force=args.force))
        except DownloadError as exc:
            LOG.error("[%s] ОШИБКА: %s", spec.name, exc)
            failed.append(spec.name)

    # Сводка.
    n_dl = sum(1 for e in entries if e["status"] == "downloaded")
    n_skip = sum(1 for e in entries if e["status"] == "skipped")
    LOG.info("Итого: скачано %d, пропущено %d, ошибок %d", n_dl, n_skip, len(failed))

    # MANIFEST пишем только для успешно обработанных моделей (частичный
    # манифест полезнее отсутствующего: офлайн-проверка покажет, чего нет).
    if entries:
        path = write_manifest(models_dir, entries)
        LOG.info("Манифест записан: %s", path)

    if failed:
        LOG.error("Не скачаны: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
