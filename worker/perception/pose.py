"""Поза людей: 17 ключевых точек COCO (ViTPose) и геометрия по ним.

Схема top-down: рамки людей берутся из сцены (cashier_bbox, others[].bbox), модель ставит
точки внутри каждой рамки. Модель — usyd-community/vitpose-base-simple (Apache-2.0,
проверено на странице модели 17.10.2026), классы VitPose* есть в transformers начиная с 4.48.

torch и transformers импортируются лениво, внутри PoseEstimator: геометрия ниже — только
stdlib, её используют признаки (worker.scene.pose_features) и тесты на машине без GPU.

Без распознавания лиц (FAQ 29): точки головы (нос, глаза, уши) — только геометрия, где
находится голова. Вырезки лиц, эмбеддинги и сравнение людей между видео не делаются.

Координаты точек — доли кадра (x, y), как во всём контракте; score — уверенность модели 0..1.
Расстояния считаются в пикселях (доля × frame_size): кадр 464×848 не квадратный.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = [
    "KEYPOINT_NAMES",
    "NUM_KEYPOINTS",
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "HEAD", "WRISTS", "SKELETON",
    "DEFAULT_MIN_SCORE",
    "POSE_MODEL",
    "Keypoints",
    "to_px",
    "dist_px",
    "visible",
    "point",
    "wrists",
    "head_point",
    "scale",
    "PoseEstimator",
    "load_pose_model",
]

LOG = logging.getLogger("worker.perception.pose")

# --------------------------------------------------------------------------- COCO-17

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
(NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
 LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
 LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE) = range(NUM_KEYPOINTS)

HEAD: tuple[int, ...] = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
WRISTS: tuple[int, ...] = (LEFT_WRIST, RIGHT_WRIST)

# Рёбра скелета — как config.edges у vitpose-base-simple (для отрисовки в scripts/viz_pose.py).
SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)

DEFAULT_MIN_SCORE = 0.3          # точка ниже этого score считается невидимой

# Перевод запасных мер в «ширину плеч»: у взрослого торс (середина плеч — середина бёдер)
# примерно в 1,3 раза длиннее ширины плеч, видимая рамка сидящего — около четырёх ширин плеч.
TORSO_TO_SHOULDERS = 0.75
BBOX_HEIGHT_TO_SHOULDERS = 0.25
# Плечи боком к камере сливаются: ширина меньше половины торсовой оценки — берём торс.
SIDEWAYS_RATIO = 0.5


@dataclass(frozen=True)
class _PoseModelRef:
    repo: str
    revision: str
    local_dir: str
    license: str


# Та же модель и ревизия, что в scripts/download_models.py (группа vitpose).
POSE_MODEL = _PoseModelRef(
    repo="usyd-community/vitpose-base-simple",
    revision="a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
    local_dir="vitpose-base-simple",
    license="Apache-2.0",
)

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Keypoints:
    """17 точек COCO одного человека: (x, y, score), x и y — доли кадра."""

    points: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) != NUM_KEYPOINTS:
            raise ValueError(f"ожидается {NUM_KEYPOINTS} точек, получено {len(self.points)}")
        for p in self.points:
            if len(p) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
                raise ValueError(f"точка должна быть (x, y, score) из конечных чисел: {p!r}")

    def to_list(self, digits: int = 3) -> list[list[float]]:
        """Для JSON сцены: [[x, y, score], ...] с округлением."""
        return [[round(float(v), digits) for v in p] for p in self.points]

    @classmethod
    def from_list(cls, raw: Any) -> Optional["Keypoints"]:
        """Из JSON сцены (None или пустое значение — None; неверная форма — ValueError)."""
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"ожидается список точек, получено {type(raw).__name__}")
        return cls(tuple((float(p[0]), float(p[1]), float(p[2])) for p in raw))


# --------------------------------------------------------------------------- геометрия (stdlib)


def to_px(p: Point, frame_size: tuple[int, int]) -> Point:
    """Доли кадра -> пиксели; frame_size = (ширина, высота)."""
    return (p[0] * frame_size[0], p[1] * frame_size[1])


def dist_px(a: Point, b: Point, frame_size: tuple[int, int]) -> float:
    """Расстояние между точками (доли кадра) в пикселях."""
    return math.hypot((a[0] - b[0]) * frame_size[0], (a[1] - b[1]) * frame_size[1])


def visible(kp: Optional[Keypoints], i: int, min_score: float = DEFAULT_MIN_SCORE) -> bool:
    return kp is not None and kp.points[i][2] >= min_score


def point(kp: Optional[Keypoints], i: int, min_score: float = DEFAULT_MIN_SCORE) -> Optional[Point]:
    """(x, y) видимой точки или None."""
    if not visible(kp, i, min_score):
        return None
    x, y, _ = kp.points[i]  # type: ignore[union-attr]
    return (x, y)


def wrists(kp: Optional[Keypoints], min_score: float = DEFAULT_MIN_SCORE) -> list[Point]:
    """Видимые запястья (0, 1 или 2 точки)."""
    return [p for p in (point(kp, i, min_score) for i in WRISTS) if p is not None]


def _mean_point(points: Sequence[Point]) -> Optional[Point]:
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def head_point(kp: Optional[Keypoints], min_score: float = DEFAULT_MIN_SCORE) -> Optional[Point]:
    """Где голова: среднее видимых точек носа, глаз и ушей (только геометрия, не лицо)."""
    return _mean_point([p for p in (point(kp, i, min_score) for i in HEAD) if p is not None])


def _pair_mean(kp: Optional[Keypoints], left: int, right: int, min_score: float) -> Optional[Point]:
    return _mean_point([p for p in (point(kp, left, min_score), point(kp, right, min_score)) if p is not None])


def scale(
    kp: Optional[Keypoints],
    frame_size: tuple[int, int],
    bbox: Optional[BBox] = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Optional[float]:
    """Масштаб человека s в пикселях — единица всех порогов позы.

    Ширина плеч; если плечи не видны (или человек боком и плечи слились) — длина торса × 0,75;
    если и её нет — высота рамки × 0,25. None — масштаб определить нечем.
    """
    shoulders: Optional[float] = None
    ls, rs = point(kp, LEFT_SHOULDER, min_score), point(kp, RIGHT_SHOULDER, min_score)
    if ls is not None and rs is not None:
        shoulders = dist_px(ls, rs, frame_size)
    # Торс — среднее плечо–бедро одной стороны (видна одна сторона — по ней); иначе середины.
    sides = [dist_px(sh, hip, frame_size) for sh, hip in (
        (ls, point(kp, LEFT_HIP, min_score)), (rs, point(kp, RIGHT_HIP, min_score)))
        if sh is not None and hip is not None]
    torso: Optional[float] = None
    if sides:
        torso = sum(sides) / len(sides) * TORSO_TO_SHOULDERS
    else:
        mid_sh = _pair_mean(kp, LEFT_SHOULDER, RIGHT_SHOULDER, min_score)
        mid_hip = _pair_mean(kp, LEFT_HIP, RIGHT_HIP, min_score)
        if mid_sh is not None and mid_hip is not None:
            torso = dist_px(mid_sh, mid_hip, frame_size) * TORSO_TO_SHOULDERS
    if shoulders is not None and shoulders > 0 and (torso is None or shoulders >= SIDEWAYS_RATIO * torso):
        return shoulders
    if torso is not None and torso > 0:
        return torso
    if bbox is not None:
        height = (bbox[3] - bbox[1]) * frame_size[1]
        if height > 0:
            return height * BBOX_HEIGHT_TO_SHOULDERS
    return None


# --------------------------------------------------------------------------- модель (torch)


def _xyxy_to_coco_px(box: BBox, width: int, height: int) -> list[float]:
    """[x1, y1, x2, y2] в долях кадра -> COCO [x, y, w, h] в пикселях (формат VitPoseImageProcessor)."""
    x1, y1 = box[0] * width, box[1] * height
    return [x1, y1, max(1.0, (box[2] - box[0]) * width), max(1.0, (box[3] - box[1]) * height)]


class PoseEstimator:
    """ViTPose (top-down): точки людей по рамкам. fp16 на CUDA, пачками.

    Процессор сам расширяет рамку (padding_factor 1,25) и подгоняет её к соотношению 192×256
    аффинным преобразованием, поэтому рамки из сцены передаются как есть.
    """

    def __init__(self, path: Path, device: str, *, batch_size: int = 32) -> None:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self._torch = torch
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = VitPoseForPoseEstimation.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    def __call__(
        self,
        images_bgr: Sequence[Any],
        boxes_per_image: Sequence[Sequence[Optional[BBox]]],
    ) -> list[list[Optional[Keypoints]]]:
        """Для каждого кадра и каждой рамки (доли кадра) — Keypoints или None (рамки нет)."""
        if len(images_bgr) != len(boxes_per_image):
            raise ValueError("images_bgr и boxes_per_image должны быть одной длины")
        result: list[list[Optional[Keypoints]]] = [[None] * len(boxes) for boxes in boxes_per_image]
        # (индекс кадра, индекс рамки) по всем кадрам; кадры без рамок в модель не идут.
        jobs = [(i, j) for i, boxes in enumerate(boxes_per_image) for j, b in enumerate(boxes)
                if b is not None and images_bgr[i] is not None]
        for start in range(0, len(jobs), self.batch_size):
            chunk = jobs[start:start + self.batch_size]
            order = list(dict.fromkeys(i for i, _ in chunk))
            images, boxes_px, slots = [], [], []
            for i in order:
                image = images_bgr[i]
                h, w = image.shape[:2]
                mine = [j for k, j in chunk if k == i]
                images.append(image[:, :, ::-1].copy())          # BGR -> RGB
                boxes_px.append([_xyxy_to_coco_px(boxes_per_image[i][j], w, h) for j in mine])  # type: ignore[arg-type]
                slots.append((i, mine, w, h))
            for (i, mine, w, h), poses in zip(slots, self._infer(images, boxes_px)):
                for j, pose in zip(mine, poses):
                    result[i][j] = _to_keypoints(pose, w, h)
        return result

    def _infer(self, images: list[Any], boxes_px: list[list[list[float]]]) -> list[list[Any]]:
        torch = self._torch
        inputs = self.processor(images, boxes=boxes_px, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
        outputs.heatmaps = outputs.heatmaps.float().cpu()
        # Без threshold: post_process возвращает все 17 точек на рамку, по рамкам в порядке boxes.
        return self.processor.post_process_pose_estimation(outputs, boxes=boxes_px)


def _to_keypoints(pose: Any, width: int, height: int) -> Optional[Keypoints]:
    """{'keypoints': (17, 2) в пикселях, 'scores': (17,)} -> Keypoints в долях кадра."""
    xy = pose["keypoints"].tolist() if hasattr(pose["keypoints"], "tolist") else pose["keypoints"]
    sc = pose["scores"].tolist() if hasattr(pose["scores"], "tolist") else pose["scores"]
    if len(xy) != NUM_KEYPOINTS or len(sc) != NUM_KEYPOINTS:
        LOG.warning("модель позы вернула %d точек вместо %d", len(xy), NUM_KEYPOINTS)
        return None
    pts = tuple(
        (min(1.0, max(0.0, float(x) / width)), min(1.0, max(0.0, float(y) / height)), min(1.0, max(0.0, float(s))))
        for (x, y), s in zip(xy, sc)
    )
    return Keypoints(pts)


_CACHE: dict[tuple[str, str], PoseEstimator] = {}


def load_pose_model(models_dir: Path, device: str) -> PoseEstimator:
    """Загрузить ViTPose один раз на процесс (отдельно от ModelBundle). Только с диска."""
    from worker.perception.models import ModelNotFoundError, resolve_device

    path = Path(models_dir) / "hf" / POSE_MODEL.local_dir
    if not (path / "config.json").is_file():
        raise ModelNotFoundError(
            f"нет весов {POSE_MODEL.repo} в {path}: python scripts/download_models.py --only vitpose")
    resolved = resolve_device(device)
    key = (str(path.resolve()), resolved)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    started = time.monotonic()
    estimator = PoseEstimator(path, resolved)
    LOG.info("модель позы загружена за %.1f с (device=%s)", time.monotonic() - started, resolved)
    _CACHE[key] = estimator
    return estimator
