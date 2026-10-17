"""Точка подключения пайплайна (контракт, раздел 11).

    analyze(video_path, job_id, work_dir, rules) -> AnalysisResult
        — единственная функция, которую вызывают worker/main.py и scripts/evaluate.py;
        — ЕДИНСТВЕННОЕ место выбора «настоящий пайплайн / мок».
    process_video(...) — настоящий пайплайн (пока NotImplementedError):
        perception.extract_frames / perception.detect -> events.detect_events -> verdict.build_result

work_dir — каталог задачи (в worker: STORAGE_DIR/<job_id>; кадры — work_dir/frames/<name>.jpg).
Без HTTP и без БД: функция чистая относительно хранилища задач.
"""

from __future__ import annotations

import logging
from pathlib import Path

from shared.schemas import AnalysisResult
from worker.events import Rule, RulesError, load_rules

__all__ = ["Rule", "RulesError", "load_rules", "process_video", "analyze"]

LOG = logging.getLogger("worker.runner")


def process_video(video_path: Path, job_id: str, work_dir: Path, rules: list[Rule]) -> AnalysisResult:
    """Настоящий пайплайн анализа видео. Пока не реализован."""
    raise NotImplementedError


def analyze(video_path: Path, job_id: str, work_dir: Path, rules: list[Rule]) -> AnalysisResult:
    """Проанализировать видео: настоящий пайплайн, а пока его нет — мок."""
    video_path, work_dir = Path(video_path), Path(work_dir)
    if not rules:
        raise ValueError("список правил пуст")
    try:
        return process_video(video_path, job_id, work_dir, rules)
    except NotImplementedError:
        # TODO(МОК): убрать ветку вместе с worker/mock.py, когда process_video будет реализован.
        from worker.mock import mock_analyze

        LOG.warning("задача %s: пайплайн не реализован, МОК-результат", job_id)
        return mock_analyze(job_id, video_path, work_dir, rules[0])
