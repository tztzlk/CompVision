"""Проход модели позы по готовой сцене: ключевые точки людей -> признаки -> поля состояний.

Заглушка с зафиксированной сигнатурой: реализацию пишет задание по позе (ветка pose-work).
Вызывается из worker.scene.pipeline.run_pipeline после assemble_timeline, только при
ENABLE_POSE=1 (по умолчанию выключено — поведение сервиса не меняется).

Новые поля состояний кладутся в SceneState.extra (правило 7 контракта): cashier_kp, others_kp
и признаки позы. Детекторы читают их через SceneState.value(name).
"""

from __future__ import annotations

from pathlib import Path

from worker.scene.contract import SceneTimeline

__all__ = ["run_pose_pass"]


def run_pose_pass(timeline: SceneTimeline, frames_dir: Path, models_dir: Path, device: str) -> SceneTimeline:
    """Добавить в состояния точки и признаки позы. Пока заглушка: сцена возвращается без изменений.

    frames_dir — каталог кадров анализа (work_dir/frames); путь кадра состояния — frames_dir.parent / state.frame.
    """
    return timeline
