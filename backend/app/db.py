"""Фасад хранилища для backend: только то, что нужно API. Весь SQL — в shared/db.py."""

from shared.db import FeedbackError, add_feedback, connect, create_job, get_job, init_db, sqlite_path_from_url

__all__ = ["connect", "init_db", "create_job", "get_job", "add_feedback", "FeedbackError", "sqlite_path_from_url"]
