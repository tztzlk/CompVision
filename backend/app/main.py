"""FastAPI-приложение backend.

Точка входа совпадает с CMD в backend/Dockerfile:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Маршруты:
    GET  /health                                  — живость (без ключа).
    POST /api/v1/analyze                          — загрузить видео (поле file), вернуть job_id. Раздел 10 ТЗ.
    GET  /api/v1/jobs/{job_id}                    — статус processing/done/failed и результат. Раздел 10 ТЗ.
    GET  /api/v1/jobs/{job_id}/files/{filename}   — файл задачи (input.mov, frames/<name>.jpg).
    POST /api/v1/jobs/{job_id}/feedback           — отзыв проверяющего по нарушению (самообучение):
         {violation_index, label: confirmed|rejected, comment?}; 404 — нет задачи, 409 — задача не готова
         или нет такого нарушения. Формат ответа GET /jobs/{job_id} не меняется.
Все /api/* требуют заголовок X-API-Key == API_KEY; иначе 401 (проверка в middleware ДО чтения тела,
OPTIONS-preflight пропускается). Затем ранний 413 по Content-Length. Не .mov/QuickTime -> 415.
Слои (снаружи внутрь): CORSMiddleware -> _guard (ключ, 413) -> маршруты.
Ошибки — JSON {"detail": "..."}.
Backend только создаёт/читает задачи и отдаёт файлы; очередь обслуживает worker.
"""

from __future__ import annotations

import hmac
import logging
import re
import shutil
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from shared.schemas import AnalysisResult, AnalyzeResponse, JobStatus

from . import db
from .config import Settings, load_cors_origins, load_settings

LOG = logging.getLogger("backend")

INPUT_NAME = "input.mov"
_CHUNK = 1024 * 1024
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
# запас на multipart-обвязку при ранней проверке Content-Length
_MULTIPART_OVERHEAD = 64 * 1024
# Контракт, раздел 10: тип бокса ISO BMFF/QuickTime в байтах 4..8
_QT_BOX_TYPES = frozenset({b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"})
_BAD_FORMAT_DETAIL = "ожидается видео .mov (QuickTime)"


class FeedbackRequest(BaseModel):
    """Тело POST /api/v1/jobs/{job_id}/feedback."""

    model_config = ConfigDict(extra="forbid")

    violation_index: int = Field(ge=0, description="Индекс в violations[] ответа GET /jobs/{job_id}.")
    label: Literal["confirmed", "rejected"]
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    status: Literal["ok"] = "ok"


class _TooLarge(Exception):
    pass


class _BadFormat(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Старт: настройки, каталог хранилища, схема БД (идемпотентно)."""
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    with closing(db.connect(settings.database_url)) as conn:
        db.init_db(conn)
    app.state.settings = settings
    LOG.info("backend запущен: storage=%s, max_upload=%d МБ", settings.storage_dir, settings.max_upload_mb)
    yield


app = FastAPI(title="CompVision — кассовая дисциплина", version="0.1.0", lifespan=lifespan)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@app.middleware("http")
async def _guard(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Для /api/: сначала X-API-Key (constant-time, до чтения тела; OPTIONS пропускается),
    затем ранний 413 по Content-Length. Точная проверка размера — при потоковой записи в analyze()."""
    path = request.url.path
    if (path == "/api" or path.startswith("/api/")) and request.method != "OPTIONS":
        settings: Settings = request.app.state.settings
        given = request.headers.get("x-api-key")
        if not given or not hmac.compare_digest(
            given.encode("utf-8"), settings.api_key.encode("utf-8")
        ):
            return JSONResponse(status_code=401, content={"detail": "неверный или отсутствующий X-API-Key"})
        if request.method == "POST" and path == "/api/v1/analyze":
            length = request.headers.get("content-length")
            if length and length.isdigit():
                if int(length) > settings.max_upload_bytes + _MULTIPART_OVERHEAD:
                    return JSONResponse(status_code=413, content={"detail": "файл превышает MAX_UPLOAD_MB"})
    return await call_next(request)


# Добавлен последним => самый внешний слой: CORS-заголовки есть и у ответов 401/413.
app.add_middleware(
    CORSMiddleware,
    allow_origins=load_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Живость процесса (healthcheck в Dockerfile/compose, nginx /health). Без ключа."""
    return {"status": "ok"}


def _looks_like_quicktime(head: bytes) -> bool:
    return len(head) >= 8 and head[4:8] in _QT_BOX_TYPES


def _copy_limited(src: BinaryIO, dst: Path, limit: int, name_is_mov: bool) -> int:
    """Потоковая запись с лимитом. Формат проверяется по первому чанку (файл не читается дважды):
    принять, если имя *.mov ИЛИ первые байты — бокс ISO BMFF/QuickTime; иначе _BadFormat."""
    written = 0
    first = True
    with dst.open("wb") as out:
        while True:
            chunk = src.read(_CHUNK)
            if first:
                first = False
                if not name_is_mov and not _looks_like_quicktime(chunk or b""):
                    raise _BadFormat()
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise _TooLarge()
            out.write(chunk)
    return written


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, file: UploadFile = File(...)) -> AnalyzeResponse:
    """Принять видео .mov/QuickTime (поле file, до MAX_UPLOAD_MB; иначе 415/413), сохранить в STORAGE_DIR/<job_id>/input.mov,
    создать задачу со статусом processing, вернуть job_id."""
    settings = _settings(request)
    job_id = uuid.uuid4().hex[:12]
    job_dir = settings.storage_dir / job_id
    name_is_mov = (file.filename or "").lower().endswith(".mov")
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        size = await run_in_threadpool(
            _copy_limited, file.file, job_dir / INPUT_NAME, settings.max_upload_bytes, name_is_mov
        )
    except _TooLarge:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail="файл превышает MAX_UPLOAD_MB")
    except _BadFormat:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=415, detail=_BAD_FORMAT_DETAIL)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    def _create() -> None:
        with closing(db.connect(settings.database_url)) as conn:
            db.create_job(conn, job_id, Path(job_id) / INPUT_NAME)

    try:
        await run_in_threadpool(_create)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    LOG.info("задача %s создана: %s, %d байт", job_id, file.filename, size)
    return AnalyzeResponse(job_id=job_id, status=JobStatus.PROCESSING)


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=AnalysisResult,
    response_model_exclude_none=True,
)
def get_job(request: Request, job_id: str) -> AnalysisResult:
    """Статус и результат задачи; 404, если job_id неизвестен."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="задача не найдена")
    with closing(db.connect(_settings(request).database_url)) as conn:
        result = db.get_job(conn, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    return result


@app.get("/api/v1/jobs/{job_id}/files/{filename:path}")
def get_job_file(request: Request, job_id: str, filename: str) -> FileResponse:
    """Файл из STORAGE_DIR/<job_id>/ (input.mov, frames/<name>.jpg). Выход за каталог задачи -> 404."""
    not_found = HTTPException(status_code=404, detail="файл не найден")
    if not _JOB_ID_RE.match(job_id) or not filename or "\x00" in filename:
        raise not_found
    base = (_settings(request).storage_dir / job_id).resolve()
    try:
        target = (base / filename).resolve()
    except (OSError, RuntimeError):
        raise not_found
    if target == base or not target.is_relative_to(base) or not target.is_file():
        raise not_found
    return FileResponse(target)


@app.post("/api/v1/jobs/{job_id}/feedback", response_model=FeedbackResponse)
def post_feedback(request: Request, job_id: str, body: FeedbackRequest) -> FeedbackResponse:
    """Подтвердить или отклонить нарушение. Отзывы копятся в таблице feedback для переобучения."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="задача не найдена")
    with closing(db.connect(_settings(request).database_url)) as conn:
        try:
            record = db.add_feedback(conn, job_id, body.violation_index, body.label, body.comment)
        except db.FeedbackError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    if record is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    LOG.info("задача %s: отзыв %s по нарушению %d (%s)", job_id, body.label, body.violation_index, record.rule_id)
    return FeedbackResponse()
