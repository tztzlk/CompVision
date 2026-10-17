"""Настройки backend из переменных окружения.

Имена переменных совпадают с docker-compose.yml и .env.example:
API_KEY, STORAGE_DIR, DATABASE_URL, MODELS_DIR, LOG_LEVEL, MAX_UPLOAD_MB, JOB_TIMEOUT_SEC.
CORS_ALLOW_ORIGINS читается отдельно (load_cors_origins): CORS-слой подключается при импорте app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    storage_dir: Path          # STORAGE_DIR=/data/storage — видео и кадры (общий volume с worker)
    database_url: str          # DATABASE_URL=sqlite:////data/app.db — общая SQLite с worker
    models_dir: Path           # MODELS_DIR=/models — веса, read-only (backend их не использует)
    log_level: str             # LOG_LEVEL=info
    max_upload_mb: int         # MAX_UPLOAD_MB=200
    job_timeout_sec: int       # JOB_TIMEOUT_SEC=300 — лимит ТЗ 5 минут (таймауты применяет worker)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip() or str(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} должно быть целым числом, получено {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} должно быть > 0, получено {value}")
    return value


def load_settings() -> Settings:
    """Прочитать Settings из os.environ с дефолтами как в Dockerfile/.env.example.

    API_KEY обязателен: без него сервис не стартует.
    """
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_KEY не задан: без ключа backend не запускается")
    if api_key.startswith("change-me") or len(api_key) < 16:
        raise RuntimeError("API_KEY — плейсхолдер или короче 16 символов: сгенерируйте `openssl rand -hex 32`")
    return Settings(
        api_key=api_key,
        storage_dir=Path(os.environ.get("STORAGE_DIR", "/data/storage")),
        database_url=os.environ.get("DATABASE_URL", "sqlite:////data/app.db"),
        models_dir=Path(os.environ.get("MODELS_DIR", "/models")),
        log_level=os.environ.get("LOG_LEVEL", "info").strip().lower() or "info",
        max_upload_mb=_positive_int("MAX_UPLOAD_MB", 200),
        job_timeout_sec=_positive_int("JOB_TIMEOUT_SEC", 300),
    )


def load_cors_origins() -> list[str]:
    """CORS_ALLOW_ORIGINS: список origin через запятую; пусто/не задано -> ["*"]."""
    raw = os.environ.get("CORS_ALLOW_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]
