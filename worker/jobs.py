"""Сторона worker в SQLite-очереди: тонкий фасад над shared.db (весь SQL — там)."""

from shared.db import (
    ClaimedJob,
    claim_next_job,
    connect,
    fail_stale,
    requeue_inflight,
    init_db,
    mark_failed,
    save_result,
)

__all__ = [
    "ClaimedJob",
    "connect",
    "init_db",
    "claim_next_job",
    "save_result",
    "mark_failed",
    "fail_stale",
    "requeue_inflight",
]
