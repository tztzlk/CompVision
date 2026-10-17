"""Слой состояний сцены: контракт SceneTimeline и build_scene (видео -> SceneTimeline).

Импорт лёгкий (только stdlib): контракт нужен events/verdict/evaluate без torch и OpenCV.
build_scene подгружает OpenCV и модели лениво (worker.scene.pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from worker.scene.builder import SceneConfig
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
    "SceneConfig",
    "build_scene",
]


def build_scene(
    video_path: Path,
    work_dir: Path,
    models_dir: Path,
    device: str,
    *,
    video_id: Optional[str] = None,
    config: Optional[SceneConfig] = None,
    raw_out: Optional[Path] = None,
) -> SceneTimeline:
    """Видео -> SceneTimeline. Кадры пишет в work_dir/frames/, модели загружает один раз на процесс.

    video_id по умолчанию — имя файла без расширения. В worker файл задачи называется input.mov,
    поэтому runner передаёт job_id (или исходное имя). device: "cuda" | "cpu" (без CUDA — cpu).
    raw_out — куда записать сырой кэш восприятия (worker.scene.raw_cache), None — не писать.
    """
    from worker.scene.pipeline import run_pipeline

    return run_pipeline(video_path, work_dir, models_dir, device, video_id=video_id, config=config, raw_out=raw_out)
