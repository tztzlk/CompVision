"""Точка подключения пайплайна (контракт, раздел 11).

    analyze(video_path, job_id, work_dir, rules) -> AnalysisResult
        — единственная функция, которую вызывают worker/main.py и scripts/evaluate.py;
    process_video(...) — пайплайн: scene.build_scene -> events.detect_events -> verdict.build_result;
    analyze_scene(...) — та же вторая половина на готовом SceneTimeline (кэш data/scene_cache,
        готовится scripts/build_scene_cache.py: подбор порогов без моделей и GPU).

work_dir — каталог задачи (в worker: STORAGE_DIR/<job_id>; кадры — work_dir/frames/<name>.jpg).
Без HTTP и без БД: функция чистая относительно хранилища задач.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from shared.schemas import AnalysisResult
from worker.events import Rule, RulesError, detect_events, load_rules
from worker.scene.contract import SceneTimeline
from worker.verdict import build_result

__all__ = ["Rule", "RulesError", "load_rules", "process_video", "analyze", "analyze_scene"]

LOG = logging.getLogger("worker.runner")


def analyze_scene(
    scene: SceneTimeline, job_id: str, work_dir: Path, rules: list[Rule], video_path: Optional[Path] = None
) -> AnalysisResult:
    """Правила и ответ по готовому временному ряду состояний."""
    events = detect_events(rules, scene)
    LOG.info("задача %s: событий до фильтра — %d", job_id, len(events))
    return build_result(job_id, events, rules, Path(work_dir), video_path)


def process_video(
    video_path: Path,
    job_id: str,
    work_dir: Path,
    rules: list[Rule],
    models_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> AnalysisResult:
    """Пайплайн анализа видео. models_dir/device по умолчанию — из worker.config.load_settings()."""
    from worker.scene import build_scene

    if models_dir is None or device is None:
        from worker.config import load_settings

        settings = load_settings()
        models_dir = settings.models_dir if models_dir is None else models_dir
        device = settings.device if device is None else device
    t0 = time.monotonic()
    # В worker файл задачи называется input.mov, поэтому video_id = job_id.
    scene = build_scene(video_path, work_dir, Path(models_dir), device, video_id=job_id)
    LOG.info("задача %s: сцена построена за %.1f с, состояний %d",
             job_id, time.monotonic() - t0, len(scene.states))
    return analyze_scene(scene, job_id, work_dir, rules, video_path)


def analyze(video_path: Path, job_id: str, work_dir: Path, rules: list[Rule]) -> AnalysisResult:
    """Проанализировать видео."""
    video_path, work_dir = Path(video_path), Path(work_dir)
    if not rules:
        raise ValueError("список правил пуст")
    return process_video(video_path, job_id, work_dir, rules)
