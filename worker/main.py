"""Точка входа worker. Совпадает с CMD в worker/Dockerfile: python -m worker.main

Цикл очереди (только здесь, контракт раздел 4):
    jobs.claim_next_job -> runner.analyze(video, job_id, STORAGE_DIR/<job_id>, rules)
    -> jobs.save_result | jobs.mark_failed
Выбор «пайплайн / мок» — только в worker/runner.py (сейчас МОК, см. worker/mock.py).
В каждом проходе цикла задачи processing, созданные раньше JOB_TIMEOUT_SEC назад, -> failed (shared.db.fail_stale).
SIGTERM/SIGINT: дорабатывает текущую задачу (subprocess ограничены таймаутами) и выходит с кодом 0.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import threading

from worker import jobs, runner
from worker.config import WorkerSettings, load_settings
from worker.events import Rule, load_rules

LOG = logging.getLogger("worker")

_stop = threading.Event()


def _handle_signal(signum: int, _frame: object) -> None:
    LOG.info("получен сигнал %s, останавливаюсь", signum)
    _stop.set()


def process_one(conn: sqlite3.Connection, settings: WorkerSettings, rules: list[Rule]) -> bool:
    """Взять и обработать одну задачу. True — задача была, False — очередь пуста."""
    job = jobs.claim_next_job(conn)
    if job is None:
        return False
    LOG.info("задача %s взята в работу", job.job_id)
    video = settings.storage_dir / job.video_path   # абсолютный video_path сохраняется как есть
    try:
        result = runner.analyze(video, job.job_id, settings.storage_dir / job.job_id, rules)
    except Exception as exc:  # noqa: BLE001 — любая ошибка задачи -> failed, worker живёт дальше
        LOG.warning("задача %s: ошибка обработки: %s", job.job_id, exc)
        jobs.mark_failed(conn, job.job_id, f"{type(exc).__name__}: {exc}")
        return True
    if jobs.save_result(conn, result):
        LOG.info("задача %s: done, verdict=%s", job.job_id, result.verdict.value)
    else:
        LOG.warning("задача %s: результат не записан (задача уже не в processing)", job.job_id)
    return True


def main() -> int:
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    # TODO(МОК): убрать предупреждение, когда runner.process_video будет реализован.
    LOG.warning("МОК-режим: перцепции нет, каждое видео получает тестовое нарушение-заглушку")

    try:
        rules = load_rules(settings.rules_path)
    except ValueError as exc:
        LOG.error("rules.yaml не загружен: %s", exc)
        return 1
    LOG.info("правил загружено: %d (%s)", len(rules), settings.rules_path)

    conn = jobs.connect(settings.database_url)
    jobs.init_db(conn)
    for job_id in jobs.requeue_inflight(conn):
        LOG.warning("задача %s была прервана (рестарт worker), возвращена в очередь", job_id)
    LOG.info("worker запущен: device=%s, poll=%.1f с, timeout=%d с",
             settings.device, settings.poll_interval_sec, settings.job_timeout_sec)
    try:
        while not _stop.is_set():
            try:
                for job_id in jobs.fail_stale(conn, settings.job_timeout_sec):
                    LOG.warning("задача %s: таймаут, переведена в failed", job_id)
                had_job = process_one(conn, settings, rules)
            except sqlite3.Error as exc:
                LOG.error("ошибка БД: %s", exc)
                had_job = False
            if not had_job:
                _stop.wait(settings.poll_interval_sec)
    finally:
        conn.close()
    LOG.info("worker остановлен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
