"""Лёгкие головы поверх замороженной SigLIP: логистическая регрессия на эмбеддинге вырезки зоны.

Веса — models/heads/<имя>.npz (scripts/train_heads.py по меткам training/labels/): w (768,), b, meta.
Инференс — numpy: sigmoid(emb @ w + b), emb — L2-нормированный get_image_features SigLIP
(ZeroShotClassifier.image_embeddings). Включается полем SceneConfig.safe_head; пусто — zero-shot как раньше.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__all__ = ["LinearHead", "load_head", "PERSON_MARGIN", "PERSON_GEO_SCALE", "person_geo", "crop_person"]

LOG = logging.getLogger("worker.perception.heads")


# Голова «человек внутри кассы» (SceneConfig.inside_head): вход — эмбеддинг вырезки рамки человека
# с полем PERSON_MARGIN и четыре числа геометрии (person_geo), умноженные на PERSON_GEO_SCALE —
# к порядку величины компонент нормированного эмбеддинга. Ширина рамки к росту — масштаб человека,
# когда ноги закрыты столом или кассиром (Х-33: посторонний за спиной кассира шире людей в коридоре).
PERSON_MARGIN = 0.1
PERSON_GEO_SCALE = (0.5, 4.0, 0.5, 0.5)


def person_geo(bbox: Any, score: float, ref_height: float, desk: Optional[Any]) -> list[float]:
    """[высота рамки / рост, низ рамки − верх стола, score, ширина рамки / рост] (доли кадра).

    Рост — worker.scene.builder.track_height по трекам видео."""
    x1, y1, x2, y2 = bbox
    return [(y2 - y1) / ref_height if ref_height > 0 else 0.0,
            (y2 - desk[1]) if desk is not None else 0.0,
            float(score),
            (x2 - x1) / ref_height if ref_height > 0 else 0.0]


def crop_person(image: Any, bbox: Any) -> Optional[Any]:
    """Вырезка рамки человека (доли кадра) с полем PERSON_MARGIN; None, если меньше 8 px."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * PERSON_MARGIN, (y2 - y1) * PERSON_MARGIN
    a, b = int(max(0, (x1 - mx) * w)), int(max(0, (y1 - my) * h))
    c, d = int(min(w, (x2 + mx) * w + 1)), int(min(h, (y2 + my) * h + 1))
    if c - a < 8 or d - b < 8:
        return None
    return image[b:d, a:c]


@dataclass(frozen=True)
class LinearHead:
    name: str
    w: Any                    # numpy (D,)
    b: float
    meta: dict[str, Any]

    def __call__(self, emb: Any) -> Any:
        """Вероятности 0..1 для эмбеддингов N×D."""
        import numpy as np

        x = np.asarray(emb, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))


def load_head(models_dir: Path, rel: str) -> Optional[LinearHead]:
    """Голова по пути rel (относительно models_dir или абсолютному); пусто — None.

    Нет файла или он битый — None с предупреждением в лог: сцена считается zero-shot, задача не падает.
    """
    if not rel:
        return None
    import numpy as np

    path = Path(rel)
    if not path.is_absolute():
        path = Path(models_dir) / path
    try:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        return LinearHead(name=path.stem, w=data["w"].astype(np.float32), b=float(data["b"]), meta=meta)
    except Exception as exc:          # нет файла, другой формат — работаем как без головы
        LOG.warning("голова %s не загружена (%s): остаётся zero-shot", path, exc)
        return None
