"""Детекторы по признакам позы (поля состояний из worker.scene.pose_pass).

Заглушка: реестр пуст, детекторы добавляет задание по позе (ветка pose-work). Реестр
подключается в конце worker/events/detectors.py: DETECTORS.update(POSE_DETECTORS).
Модуль импортируется из detectors.py после всех его определений, поэтому здесь можно
импортировать общую механику: from worker.events.detectors import _events_from_series.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from worker.events.model import Event
from worker.scene.contract import SceneTimeline

__all__ = ["POSE_DETECTORS"]

POSE_DETECTORS: dict[str, Callable[[SceneTimeline, Mapping[str, Any]], list[Event]]] = {}
