"""SQLite-хранилище задач, общее для backend и worker. Весь SQL проекта — только здесь.

Файл БД: DATABASE_URL=sqlite:////data/app.db на общем volume app-data.
backend: init_db, create_job, get_job, add_feedback (через фасад backend/app/db.py).
обучение (scripts/retrain_from_feedback.py): list_feedback.
worker:  init_db, claim_next_job, save_result, mark_failed, fail_stale (через worker/jobs.py).

Соединения в режиме autocommit (isolation_level=None), WAL, busy_timeout=5000:
два процесса в разных контейнерах пишут в один файл. Каждая изменяющая операция —
один SQL-оператор, т.е. атомарна сама по себе.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from shared.schemas import AnalysisResult, JobStatus

__all__ = [
    "ClaimedJob",
    "sqlite_path_from_url",
    "connect",
    "init_db",
    "create_job",
    "get_job",
    "claim_next_job",
    "save_result",
    "mark_failed",
    "fail_stale",
    "requeue_inflight",
    "FeedbackError",
    "FeedbackRecord",
    "add_feedback",
    "list_feedback",
]

LOG = logging.getLogger("shared.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,            -- shared.schemas.JobStatus
    video_path   TEXT NOT NULL,            -- путь к видео относительно STORAGE_DIR (или абсолютный)
    result_json  TEXT,                     -- AnalysisResult.model_dump_json() при done
    error        TEXT,                     -- причина при failed (наружу не отдаётся)
    created_at   REAL NOT NULL,
    started_at   REAL,                     -- когда worker взял задачу
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs (status, started_at, created_at);
-- Обратная связь проверяющего по каждому нарушению (самообучение). Формат ответа жюри не меняется.
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(job_id),
    violation_index  INTEGER NOT NULL,     -- индекс в violations[] ответа GET /jobs/{job_id}
    rule_id          TEXT,                 -- violations[i].rule.id (NULL, если правила нет)
    timecode_sec     REAL NOT NULL,        -- violations[i].timecode_sec
    label            TEXT NOT NULL,        -- confirmed | rejected
    comment          TEXT,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_job_idx ON feedback (job_id);
"""

FEEDBACK_LABELS = ("confirmed", "rejected")
_MAX_COMMENT_LEN = 1000

_SQLITE_PREFIX = "sqlite:///"
_MAX_ERROR_LEN = 2000


class FeedbackError(ValueError):
    """Отзыв нельзя принять: задача не готова или нет нарушения с таким индексом."""


@dataclass(frozen=True)
class FeedbackRecord:
    """Строка таблицы feedback."""

    job_id: str
    violation_index: int
    rule_id: Optional[str]
    timecode_sec: float
    label: str
    comment: Optional[str]
    created_at: float


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    video_path: Path


def sqlite_path_from_url(database_url: str) -> Path:
    """sqlite:////data/app.db -> Path('/data/app.db'); sqlite:///rel.db -> Path('rel.db')."""
    if not database_url.startswith(_SQLITE_PREFIX):
        raise ValueError(f"поддерживается только sqlite:///..., получено: {database_url!r}")
    rest = database_url[len(_SQLITE_PREFIX):]
    if not rest or rest == "/":
        raise ValueError(f"в DATABASE_URL не указан путь к файлу: {database_url!r}")
    return Path(rest)


def connect(database_url: str) -> sqlite3.Connection:
    """Открыть соединение (autocommit, WAL, busy_timeout), создать каталог файла при необходимости."""
    path = sqlite_path_from_url(database_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Создать таблицу jobs, если её нет. Идемпотентно."""
    conn.executescript(_SCHEMA)


def create_job(conn: sqlite3.Connection, job_id: str, video_path: Union[Path, str]) -> None:
    """Вставить новую задачу со статусом processing."""
    conn.execute(
        "INSERT INTO jobs (job_id, status, video_path, created_at) VALUES (?, ?, ?, ?)",
        (job_id, JobStatus.PROCESSING.value, str(video_path), time.time()),
    )


def get_job(conn: sqlite3.Connection, job_id: str) -> Optional[AnalysisResult]:
    """Текущее состояние задачи в формате API или None, если задачи нет.

    failed -> только job_id и status (причина наружу не отдаётся).
    """
    row = conn.execute(
        "SELECT job_id, status, result_json FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        return None
    status = row["status"]
    if status == JobStatus.DONE.value and row["result_json"]:
        try:
            return AnalysisResult.model_validate_json(row["result_json"])
        except ValidationError:
            LOG.exception("повреждён result_json у задачи %s", job_id)
            return AnalysisResult(job_id=row["job_id"], status=JobStatus.FAILED)
    if status == JobStatus.PROCESSING.value:
        return AnalysisResult(job_id=row["job_id"], status=JobStatus.PROCESSING)
    return AnalysisResult(job_id=row["job_id"], status=JobStatus.FAILED)


def claim_next_job(conn: sqlite3.Connection) -> Optional[ClaimedJob]:
    """Атомарно взять самую старую processing-задачу без started_at и проставить started_at.

    Один оператор UPDATE ... RETURNING (SQLite >= 3.35): два worker'а не возьмут одну задачу.
    """
    row = conn.execute(
        """
        UPDATE jobs SET started_at = ?
        WHERE job_id = (
            SELECT job_id FROM jobs
            WHERE status = ? AND started_at IS NULL
            ORDER BY created_at LIMIT 1
        ) AND started_at IS NULL
        RETURNING job_id, video_path
        """,
        (time.time(), JobStatus.PROCESSING.value),
    ).fetchone()
    if row is None:
        return None
    return ClaimedJob(job_id=row["job_id"], video_path=Path(row["video_path"]))


def save_result(conn: sqlite3.Connection, result: AnalysisResult) -> bool:
    """Записать результат (status=done) в result_json.

    Пишет, только если задача ещё в processing (не переведена в failed по таймауту).
    Возвращает True, если результат записан.
    """
    if result.status is not JobStatus.DONE:
        raise ValueError("save_result принимает только результат со status=done")
    cur = conn.execute(
        "UPDATE jobs SET status = ?, result_json = ?, error = NULL, finished_at = ? "
        "WHERE job_id = ? AND status = ?",
        (
            JobStatus.DONE.value,
            result.model_dump_json(exclude_none=True),
            time.time(),
            result.job_id,
            JobStatus.PROCESSING.value,
        ),
    )
    return cur.rowcount == 1


def mark_failed(conn: sqlite3.Connection, job_id: str, error: str) -> None:
    """Перевести задачу в failed с внутренней причиной."""
    conn.execute(
        "UPDATE jobs SET status = ?, error = ?, result_json = NULL, finished_at = ? WHERE job_id = ?",
        (JobStatus.FAILED.value, (error or "")[:_MAX_ERROR_LEN], time.time(), job_id),
    )


def requeue_inflight(conn: sqlite3.Connection) -> list[str]:
    """Вернуть в очередь задачи, взятые в работу, но не завершённые (started_at сбрасывается).

    Вызывать ОДИН раз при старте worker (worker один): задача, прерванная рестартом или
    падением контейнера, иначе висела бы в processing до таймаута.
    """
    rows = conn.execute(
        "UPDATE jobs SET started_at = NULL WHERE status = ? AND started_at IS NOT NULL "
        "RETURNING job_id",
        (JobStatus.PROCESSING.value,),
    ).fetchall()
    return [r["job_id"] for r in rows]


def fail_stale(conn: sqlite3.Connection, timeout_sec: float) -> list[str]:
    """Задачи processing, СОЗДАННЫЕ (загруженные) раньше чем timeout_sec назад, -> failed.

    Отсчёт от created_at, а не от started_at: лимит жюри (5 минут) идёт с момента загрузки,
    поэтому ожидание в очереди тоже расходует бюджет. Возвращает job_id переведённых задач.
    """
    now = time.time()
    rows = conn.execute(
        "UPDATE jobs SET status = ?, error = ?, finished_at = ? "
        "WHERE status = ? AND created_at < ? "
        "RETURNING job_id",
        (
            JobStatus.FAILED.value,
            f"таймаут обработки: дольше {timeout_sec:g} с",
            now,
            JobStatus.PROCESSING.value,
            now - timeout_sec,
        ),
    ).fetchall()
    return [r["job_id"] for r in rows]


def add_feedback(
    conn: sqlite3.Connection, job_id: str, violation_index: int, label: str, comment: Optional[str] = None
) -> Optional[FeedbackRecord]:
    """Сохранить отзыв по нарушению violations[violation_index] задачи job_id.

    None — задачи нет. FeedbackError — задача не в done, неверный индекс или метка.
    rule_id и timecode_sec берутся из сохранённого результата, а не от клиента.
    """
    if label not in FEEDBACK_LABELS:
        raise FeedbackError(f"label должен быть одним из {FEEDBACK_LABELS}")
    result = get_job(conn, job_id)
    if result is None:
        return None
    if result.status is not JobStatus.DONE:
        raise FeedbackError("задача ещё не обработана")
    if not 0 <= violation_index < len(result.violations):
        raise FeedbackError(f"нет нарушения с индексом {violation_index}")
    violation = result.violations[violation_index]
    record = FeedbackRecord(
        job_id=job_id,
        violation_index=violation_index,
        rule_id=violation.rule.id if violation.rule else None,
        timecode_sec=violation.timecode_sec,
        label=label,
        comment=(comment or "").strip()[:_MAX_COMMENT_LEN] or None,
        created_at=time.time(),
    )
    conn.execute(
        "INSERT INTO feedback (job_id, violation_index, rule_id, timecode_sec, label, comment, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (record.job_id, record.violation_index, record.rule_id, record.timecode_sec,
         record.label, record.comment, record.created_at),
    )
    return record


def list_feedback(conn: sqlite3.Connection, since: float = 0.0) -> list[FeedbackRecord]:
    """Все отзывы с created_at >= since, по времени (для переобучения порогов)."""
    rows = conn.execute(
        "SELECT job_id, violation_index, rule_id, timecode_sec, label, comment, created_at "
        "FROM feedback WHERE created_at >= ? ORDER BY created_at",
        (since,),
    ).fetchall()
    return [FeedbackRecord(**dict(r)) for r in rows]
