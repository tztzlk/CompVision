"""Настройки worker из переменных окружения (имена как в docker-compose.yml).

STORAGE_DIR, DATABASE_URL, MODELS_DIR, RULES_PATH, WORKER_DEVICE, LOG_LEVEL,
JOB_TIMEOUT_SEC, WORKER_POLL_SEC.
HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 выставлены в окружении: сеть в runtime не используется.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSettings:
    storage_dir: Path       # /data/storage — общий volume с backend
    database_url: str       # sqlite:////data/app.db — общий volume с backend
    models_dir: Path        # /models — read-only, веса из scripts/download_models.py
    rules_path: Path        # /app/rules.yaml — монтируется read-only из корня репозитория
    device: str             # WORKER_DEVICE: cuda | cpu
    log_level: str
    job_timeout_sec: int    # JOB_TIMEOUT_SEC=300
    poll_interval_sec: float = 1.0


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def load_settings() -> WorkerSettings:
    """Прочитать WorkerSettings из os.environ с дефолтами как в worker/Dockerfile."""
    try:
        timeout = int(_env("JOB_TIMEOUT_SEC", "300"))
        poll = float(_env("WORKER_POLL_SEC", "1.0"))
    except ValueError as exc:
        raise RuntimeError(f"некорректное числовое значение в окружении: {exc}") from exc
    if timeout <= 0:
        raise RuntimeError("JOB_TIMEOUT_SEC должно быть > 0")
    if poll <= 0:
        raise RuntimeError("WORKER_POLL_SEC должно быть > 0")
    device = _env("WORKER_DEVICE", "cuda").lower()
    if device not in ("cuda", "cpu"):
        raise RuntimeError(f"WORKER_DEVICE должно быть cuda или cpu, получено {device!r}")
    return WorkerSettings(
        storage_dir=Path(_env("STORAGE_DIR", "/data/storage")),
        database_url=_env("DATABASE_URL", "sqlite:////data/app.db"),
        models_dir=Path(_env("MODELS_DIR", "/models")),
        rules_path=Path(_env("RULES_PATH", "/app/rules.yaml")),
        device=device,
        log_level=_env("LOG_LEVEL", "info").lower(),
        job_timeout_sec=timeout,
        poll_interval_sec=poll,
    )
