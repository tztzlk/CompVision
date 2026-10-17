"""Точка входа worker. Совпадает с CMD в worker/Dockerfile: python -m worker.main

Цикл очереди (только здесь, контракт раздел 4):
    jobs.claim_next_job -> runner.analyze(video, job_id, STORAGE_DIR/<job_id>, rules)
    -> jobs.save_result | jobs.mark_failed
Пайплайн (scene -> events -> verdict) — в worker/runner.py.
В каждом проходе цикла задачи processing, созданные раньше JOB_TIMEOUT_SEC назад, -> failed (shared.db.fail_stale).
Сторож задачи: если runner.analyze не вернулся за JOB_TIMEOUT_SEC (+ запас), задача -> failed,
а процесс завершается (os._exit(1)); compose перезапускает worker (restart: unless-stopped), очередь
не блокируется зависшей задачей. Модели загружаются при старте (одна загрузка на процесс).
SIGTERM/SIGINT: дорабатывает текущую задачу (subprocess ограничены таймаутами) и выходит с кодом 0.
"""

from __future__ import annotations

import logging
import os
import signal
import sqlite3
import threading

from worker import jobs, runner
from worker.config import WorkerSettings, load_settings
from worker.events import Rule, load_rules

LOG = logging.getLogger("worker")

_stop = threading.Event()

# Запас сверх JOB_TIMEOUT_SEC до аварийного выхода: обычный путь — fail_stale/save_result.
WATCHDOG_GRACE_SEC = 30.0


def _handle_signal(signum: int, _frame: object) -> None:
    LOG.info("получен сигнал %s, останавливаюсь", signum)
    _stop.set()


def _on_job_timeout(settings: WorkerSettings, job_id: str) -> None:
    """Сторож: задача зависла дольше лимита -> failed и выход процесса (поток анализа не прервать)."""
    LOG.error("задача %s: обработка дольше %d с, failed и перезапуск worker", job_id, settings.job_timeout_sec)
    try:
        conn = jobs.connect(settings.database_url)   # отдельное соединение: sqlite3 привязан к потоку
        try:
            jobs.mark_failed(conn, job_id, f"таймаут обработки: дольше {settings.job_timeout_sec} с")
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        LOG.exception("задача %s: не удалось отметить failed", job_id)
    logging.shutdown()
    os._exit(1)


def _free_gpu_memory() -> None:
    """Вернуть кэш CUDA после ошибки (например, OOM), чтобы следующая задача не упала так же."""
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def warm_up(settings: WorkerSettings) -> None:
    """Загрузить модели при старте: первая задача не тратит время на загрузку, нехватка весов
    видна в логе сразу. Ошибка не роняет worker — задачи получат failed с той же причиной."""
    try:
        from worker.perception.models import load_models

        load_models(settings.models_dir, settings.device)
    except Exception as exc:  # noqa: BLE001
        LOG.error("модели не загружены при старте: %s: %s", type(exc).__name__, exc)


def process_one(conn: sqlite3.Connection, settings: WorkerSettings, rules: list[Rule]) -> bool:
    """Взять и обработать одну задачу. True — задача была, False — очередь пуста."""
    job = jobs.claim_next_job(conn)
    if job is None:
        return False
    LOG.info("задача %s взята в работу", job.job_id)
    video = settings.storage_dir / job.video_path   # абсолютный video_path сохраняется как есть
    watchdog = threading.Timer(settings.job_timeout_sec + WATCHDOG_GRACE_SEC, _on_job_timeout,
                               args=(settings, job.job_id))
    watchdog.daemon = True
    watchdog.start()
    try:
        result = runner.analyze(video, job.job_id, settings.storage_dir / job.job_id, rules)
    except Exception as exc:  # noqa: BLE001 — любая ошибка задачи -> failed, worker живёт дальше
        LOG.warning("задача %s: ошибка обработки: %s", job.job_id, exc)
        jobs.mark_failed(conn, job.job_id, f"{type(exc).__name__}: {exc}")
        _free_gpu_memory()
        return True
    finally:
        watchdog.cancel()
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
    warm_up(settings)
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
