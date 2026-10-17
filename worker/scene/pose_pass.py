"""Проход модели позы по готовой сцене: ключевые точки людей -> признаки -> поля состояний.

Вызывается из worker.scene.pipeline.run_pipeline после assemble_timeline, только при
ENABLE_POSE=1 (по умолчанию выключено — поведение сервиса не меняется). Тот же add_pose
использует scripts/add_pose_to_cache.py, чтобы добавить позу в готовый кэш сцен без
перезапуска остальных моделей.

Новые поля состояний кладутся в SceneState.extra (правило 7 контракта), детекторы читают их
через SceneState.value(name):
  cashier_kp          — 17 точек кассира [[x, y, score], ...] (доли кадра, 3 знака) или null;
  others_kp           — [{track_id, kp}] для людей из others (люди за стеклом в others не входят);
  признаки позы       — worker.scene.pose_features.FEATURE_NAMES (0..1 или null);
  hand_to_hand_track  — track_id человека, с чьей кистью сблизилась кисть кассира, или null;
  <признак>_head      — вероятность головы жеста (только с PoseConfig.gesture_heads / POSE_GESTURE_HEADS;
                        тогда и сам признак — после гибрида, worker.scene.pose_heads).
В meta сцены — meta.pose: модель, шаг, число состояний с точками кассира, время прохода, режимы голов.

Поза — вспомогательный сигнал: если модель не загрузилась или упала, сцена возвращается без
полей позы (с meta.pose.error), задача не падает.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from worker.perception.pose import POSE_MODEL, Keypoints
from worker.scene.contract import SceneState, SceneTimeline
from worker.scene.pose_features import PersonPose, PoseConfig, PoseFrame, compute_features
from worker.scene.pose_heads import GestureHead, apply_gesture_heads, load_gesture_heads

__all__ = ["add_pose", "run_pose_pass", "pose_frames_from_scene", "load_heads", "CHUNK_STATES"]

LOG = logging.getLogger("worker.scene.pose")

CHUNK_STATES = 16          # состояний за один вызов модели (рамок — до ~3 на состояние)

# Кадры состояний: список (по индексу состояния) или функция index -> кадр BGR | None.
FrameSource = Union[Sequence[Optional[Any]], Callable[[int], Optional[Any]]]
Estimator = Callable[[Sequence[Any], Sequence[Sequence[Any]]], list[list[Optional[Keypoints]]]]


def _frame(frames: FrameSource, i: int) -> Optional[Any]:
    return frames(i) if callable(frames) else frames[i]


def _kp_or_none(raw: Any) -> Optional[Keypoints]:
    try:
        return Keypoints.from_list(raw)
    except (ValueError, TypeError, IndexError):
        return None


def pose_frames_from_scene(scene: SceneTimeline) -> list[PoseFrame]:
    """PoseFrame по уже записанным в сцену точкам (cashier_kp, others_kp) — пересчёт признаков без модели."""
    out: list[PoseFrame] = []
    for s in scene.states:
        by_track = {}
        for item in s.value("others_kp") or []:
            if isinstance(item, dict) and isinstance(item.get("track_id"), int):
                by_track[item["track_id"]] = _kp_or_none(item.get("kp"))
        out.append(PoseFrame(
            t=s.t,
            cashier_kp=_kp_or_none(s.value("cashier_kp")),
            cashier_bbox=s.cashier_bbox,
            others=tuple(PersonPose(p.track_id, p.bbox, by_track.get(p.track_id)) for p in s.others),
        ))
    return out


def _estimate(
    scene: SceneTimeline, frames: FrameSource, estimator: Estimator, cfg: PoseConfig,
) -> list[PoseFrame]:
    states = scene.states
    picked = [i for i in range(len(states)) if i % max(1, cfg.every_k) == 0]
    kps: dict[int, list[Optional[Keypoints]]] = {}
    for start in range(0, len(picked), CHUNK_STATES):
        chunk = picked[start:start + CHUNK_STATES]
        images, boxes, owners = [], [], []
        for i in chunk:
            image = _frame(frames, i)
            if image is None:
                continue
            st = states[i]
            images.append(image)
            boxes.append([st.cashier_bbox, *(p.bbox for p in st.others)])
            owners.append(i)
        if not images:
            continue
        for i, result in zip(owners, estimator(images, boxes)):
            kps[i] = result
    out: list[PoseFrame] = []
    for i, st in enumerate(states):
        result = kps.get(i)
        if result is None:
            out.append(PoseFrame(t=st.t, cashier_bbox=st.cashier_bbox,
                                 others=tuple(PersonPose(p.track_id, p.bbox, None) for p in st.others)))
            continue
        out.append(PoseFrame(
            t=st.t,
            cashier_kp=result[0],
            cashier_bbox=st.cashier_bbox,
            others=tuple(PersonPose(p.track_id, p.bbox, kp) for p, kp in zip(st.others, result[1:])),
        ))
    return out


def _with_pose(state: SceneState, frame: PoseFrame, features: dict[str, Any]) -> SceneState:
    extra = {k: v for k, v in state.extra.items() if not k.endswith("_head")}   # головы прошлого прохода
    extra["cashier_kp"] = frame.cashier_kp.to_list() if frame.cashier_kp is not None else None
    extra["others_kp"] = [{"track_id": p.track_id, "kp": p.kp.to_list() if p.kp is not None else None}
                          for p in frame.others]
    extra.update(features)
    return dataclasses.replace(state, extra=extra)


def add_pose(
    scene: SceneTimeline,
    frames: FrameSource,
    estimator: Estimator,
    cfg: PoseConfig = PoseConfig(),
    heads: Optional[dict[str, GestureHead]] = None,
) -> SceneTimeline:
    """Точки и признаки позы во все состояния сцены.

    frames — кадр BGR на каждое состояние (список по индексу состояния или функция index -> кадр);
    estimator — PoseEstimator или его подмена в тестах: (кадры, рамки на кадр) -> точки на рамку.
    Первая рамка кадра — кассир (cashier_bbox, может быть None), дальше — others по порядку.
    heads — головы жестов (load_heads); None или пусто — только геометрия, как без голов.
    """
    started = time.monotonic()
    pose_frames = _estimate(scene, frames, estimator, cfg)
    features = compute_features(pose_frames, scene.zones, scene.frame_size, cfg)
    if heads:
        features = apply_gesture_heads(pose_frames, features, scene.zones, scene.frame_size, heads)
    states = tuple(_with_pose(s, f, feat) for s, f, feat in zip(scene.states, pose_frames, features))
    meta_pose: dict[str, Any] = {
        "model": f"{POSE_MODEL.repo}@{POSE_MODEL.revision[:7]}",
        "every_k": cfg.every_k,
        "states_with_cashier_kp": sum(1 for f in pose_frames if f.cashier_kp is not None),
        "elapsed_sec": round(time.monotonic() - started, 2),
    }
    if heads:
        meta_pose["gesture_heads"] = {g: h.mode for g, h in sorted(heads.items())}
    return dataclasses.replace(scene, states=states, meta={**scene.meta, "pose": meta_pose})


def load_heads(cfg: PoseConfig, models_dir: Path) -> tuple[dict[str, GestureHead], Optional[str]]:
    """Головы жестов по PoseConfig.gesture_heads: (головы, ошибка). Пустой путь — ({}, None).

    Голова — вспомогательный сигнал: не загрузилась — проход идёт на одной геометрии, ошибка — в meta.
    """
    if not cfg.gesture_heads:
        return {}, None
    path = Path(cfg.gesture_heads)
    if not path.is_absolute():
        path = Path(models_dir) / path
    try:
        heads = load_gesture_heads(path)
    except Exception as exc:  # noqa: BLE001 — нет файла, другая версия признаков, битый .npz
        LOG.error("головы жестов не загружены (%s): %s", path, exc)
        return {}, f"heads: {exc}"
    LOG.info("головы жестов %s: %s", path, {g: h.mode for g, h in heads.items()})
    return heads, None


def _config_from_env() -> PoseConfig:
    raw = os.environ.get("POSE_EVERY_K", "").strip()
    try:
        every_k = max(1, int(raw)) if raw else 1
    except ValueError:
        LOG.warning("POSE_EVERY_K=%r — не целое число, беру 1", raw)
        every_k = 1
    return PoseConfig(every_k=every_k, gesture_heads=os.environ.get("POSE_GESTURE_HEADS", "").strip())


def run_pose_pass(timeline: SceneTimeline, frames_dir: Path, models_dir: Path, device: str) -> SceneTimeline:
    """Добавить в состояния точки и признаки позы (ENABLE_POSE=1 в worker.scene.pipeline).

    frames_dir — каталог кадров анализа (work_dir/frames); путь кадра состояния — frames_dir.parent / state.frame.
    Шаг по состояниям — POSE_EVERY_K (по умолчанию каждое состояние); головы жестов — POSE_GESTURE_HEADS
    (путь к .npz относительно models_dir; по умолчанию пусто — только геометрия).
    """
    from worker.perception.pose import load_pose_model
    from worker.perception.video import imread_unicode

    base = Path(frames_dir).parent
    try:
        estimator = load_pose_model(Path(models_dir), device)
    except Exception as exc:   # нет весов, старая версия transformers (нет VitPose*), нет CUDA...
        LOG.error("поза пропущена: модель не загрузилась: %s", exc)
        return dataclasses.replace(timeline, meta={**timeline.meta, "pose": {"error": f"load: {exc}"}})
    cfg = _config_from_env()
    heads, heads_error = load_heads(cfg, Path(models_dir))
    try:
        scene = add_pose(timeline, lambda i: imread_unicode(base / timeline.states[i].frame),
                         estimator, cfg, heads)
    except Exception as exc:
        LOG.exception("поза пропущена: ошибка прохода")
        return dataclasses.replace(timeline, meta={**timeline.meta, "pose": {"error": f"run: {exc}"}})
    if heads_error:
        scene = dataclasses.replace(scene, meta={**scene.meta, "pose": {**scene.meta.get("pose", {}),
                                                                        "heads_error": heads_error}})
    LOG.info("поза %s: %s", scene.video_id, scene.meta.get("pose"))
    return scene
