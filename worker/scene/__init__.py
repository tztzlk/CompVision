"""Слой состояний сцены: контракт SceneTimeline и build_scene (видео -> SceneTimeline).

Импорт лёгкий (только stdlib): контракт нужен events/verdict/evaluate без torch и OpenCV.
"""

from __future__ import annotations

from pathlib import Path

from worker.scene.contract import (
    CONTRACT_VERSION,
    FLAG_FIELDS,
    PROB_FIELDS,
    ZONE_NAMES,
    BBox,
    Person,
    SceneContractError,
    SceneState,
    SceneTimeline,
    load_scene,
    save_scene,
    scene_from_dict,
    scene_to_dict,
)

__all__ = [
    "CONTRACT_VERSION",
    "FLAG_FIELDS",
    "PROB_FIELDS",
    "ZONE_NAMES",
    "BBox",
    "Person",
    "SceneContractError",
    "SceneState",
    "SceneTimeline",
    "load_scene",
    "save_scene",
    "scene_from_dict",
    "scene_to_dict",
    "build_scene",
]


def build_scene(video_path: Path, work_dir: Path, models_dir: Path, device: str) -> SceneTimeline:
    """Видео -> SceneTimeline. Кадры пишет в work_dir/frames/, модели загружает один раз на процесс."""
    raise NotImplementedError
