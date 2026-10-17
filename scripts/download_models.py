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
    python scripts/download_models.py                 # скачать всё, что ещё не скачано
    python scripts/download_models.py --list          # показать список моделей и выйти
    python scripts/download_models.py --only yolo11n  # скачать только указанные (флаг повторяемый)
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
    """

    name: str
    url: str
    dest: str
    sha256: Optional[str] = None
    size_mb: Optional[float] = None


# ---------------------------------------------------------------------------
# Список моделей.
#
# Пример заполнения (веса YOLO по прямой ссылке на GitHub Releases):
#
#   MODELS: list[ModelSpec] = [
#       ModelSpec(
#           name="yolo11n",
#           url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
#           dest="yolo/yolo11n.pt",
#           # sha256 получить один раз после скачивания:
#           #   sha256sum models/yolo/yolo11n.pt
#           # либо взять из MANIFEST.json, который пишет этот скрипт.
#           sha256="<hex-строка из 64 символов>",
#           size_mb=5.6,
#       ),
#   ]
#
# Пример для Hugging Face (снапшот репозитория целиком, а не одного файла).
# Требует ``pip install huggingface_hub`` — импорт НЕ делать обязательным,
# поскольку в контейнере выставлен HF_HUB_OFFLINE=1 и сеть недоступна.
# Вызывать только на хосте, один раз, например так:
#
#   def download_hf_snapshot(models_dir: Path) -> None:
#       from huggingface_hub import snapshot_download  # опциональная зависимость
#       snapshot_download(
#           repo_id="openai/clip-vit-base-patch32",
#           local_dir=models_dir / "hf" / "clip-vit-base-patch32",
#           allow_patterns=["*.json", "*.safetensors", "*.txt"],
#       )
#   # В контейнере затем: HF_HUB_OFFLINE=1 и загрузка по локальному пути
#   # ``models/hf/clip-vit-base-patch32`` (from_pretrained(local_path)).
#   # Анализ по ТЗ строго визуальный — аудио-модели не нужны.
# ---------------------------------------------------------------------------
MODELS: list[ModelSpec] = []  # TODO: добавить модели


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
    """Отфильтровать по ``--only``; неизвестные имена — ошибка."""
    specs = list(specs)
    if not only:
        return specs
    by_name = {s.name: s for s in specs}
    unknown = [n for n in only if n not in by_name]
    if unknown:
        raise SystemExit(f"Неизвестные модели в --only: {', '.join(unknown)}. "
                         f"Доступны: {', '.join(by_name) or '(список пуст)'}")
    return [by_name[n] for n in only]


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
