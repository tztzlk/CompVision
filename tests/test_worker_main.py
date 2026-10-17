"""worker.main: ошибка задачи -> failed, зависшая задача -> failed по сторожу JOB_TIMEOUT_SEC."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from shared import db
from shared.schemas import JobStatus
from worker import main as wmain
from worker.config import WorkerSettings


def _settings(tmp_path: Path, timeout: int = 300) -> WorkerSettings:
    return WorkerSettings(
        storage_dir=tmp_path / "storage", database_url=f"sqlite:///{tmp_path / 'app.db'}",
        models_dir=tmp_path / "models", rules_path=tmp_path / "rules.yaml", device="cpu",
        log_level="info", job_timeout_sec=timeout,
    )


def _queue_job(settings: WorkerSettings, job_id: str = "abcdef012345"):
    conn = db.connect(settings.database_url)
    db.init_db(conn)
    db.create_job(conn, job_id, Path(job_id) / "input.mov")
    return conn


def test_error_marks_failed(tmp_path, monkeypatch, rules):
    settings = _settings(tmp_path)
    conn = _queue_job(settings)

    def boom(*args, **kwargs):
        raise RuntimeError("битое видео")

    monkeypatch.setattr(wmain.runner, "analyze", boom)
    assert wmain.process_one(conn, settings, rules) is True
    assert db.get_job(conn, "abcdef012345").status is JobStatus.FAILED
    assert wmain.process_one(conn, settings, rules) is False


def test_hung_job_is_failed_by_watchdog(tmp_path, monkeypatch, rules):
    settings = _settings(tmp_path, timeout=1)
    _queue_job(settings).close()
    monkeypatch.setattr(wmain, "WATCHDOG_GRACE_SEC", 0.0)
    exited = threading.Event()
    monkeypatch.setattr(wmain.os, "_exit", lambda code: exited.set())
    monkeypatch.setattr(wmain.logging, "shutdown", lambda: None)
    release = threading.Event()

    def hang(*args, **kwargs):
        release.wait(10)
        raise RuntimeError("прервано тестом")

    monkeypatch.setattr(wmain.runner, "analyze", hang)
    def run() -> None:   # sqlite3-соединение привязано к потоку: открываем внутри
        conn = db.connect(settings.database_url)
        try:
            wmain.process_one(conn, settings, rules)
        finally:
            conn.close()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert exited.wait(5), "сторож не сработал"
        check = db.connect(settings.database_url)
        assert db.get_job(check, "abcdef012345").status is JobStatus.FAILED
        check.close()
    finally:
        release.set()
        worker.join(5)


@pytest.mark.parametrize("fails", [False, True])
def test_warm_up_never_raises(tmp_path, monkeypatch, fails):
    import worker.perception.models as models

    calls = []

    def fake_load(models_dir, device):
        calls.append((models_dir, device))
        if fails:
            raise models.ModelNotFoundError("нет весов")

    monkeypatch.setattr(models, "load_models", fake_load)
    wmain.warm_up(_settings(tmp_path))
    assert calls == [(tmp_path / "models", "cpu")]
