"""Дообученная визуальная башня SigLIP для голов состояний зон (scripts/finetune_siglip.py, раунд 4).

Отдельная от общей ZeroShotClassifier модель: база — те же локальные веса google/siglip-base-patch16-224
(<models_dir>/hf/siglip-base-patch16-224, local_files_only), поверх — state.pt из каталога весов
(models/ft/siglip_states/): последние N блоков энкодера, post_layernorm, голова attention pooling и
линейная многозадачная голова cls (fp16), meta.json — задачи, число блоков, метки, фолды.
Общий SigLIP (zero-shot, image_embeddings для голов раунда 3) не подменяется.

Эмбеддинг — pooler_output vision_model (= get_image_features), L2-нормированный, как при обучении;
вероятность задачи — sigmoid(emb @ cls.weight[k] + cls.bias[k]). Предобработка — тот же процессор
SigLIP, что у конвейера (вырезка зоны BGR -> RGB -> SiglipImageProcessor). fp16 на GPU, float32 на CPU.
Загружается один раз на процесс (кэш по каталогу весов и устройству), ~190 МБ в fp16.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = ["FT_STATE", "FT_META", "FinetunedTower", "load_ft_tower", "ft_tower_info"]

LOG = logging.getLogger("worker.perception.ft_tower")

FT_STATE = "state.pt"
FT_META = "meta.json"
_VM_PREFIX = "vm."          # ключи state.pt: vm.<параметр vision_model>, cls.weight, cls.bias


@dataclass
class FinetunedTower:
    name: str
    model: Any                          # transformers SiglipVisionModel
    processor: Any                      # SiglipImageProcessor (тот же, что у ZeroShotClassifier)
    weight: Any                         # torch (T, D) float32 на устройстве
    bias: Any                           # torch (T,)
    tasks: dict[str, int]               # имя задачи -> строка cls (только обученные задачи из meta.tasks)
    zones: dict[str, str]               # имя задачи -> зона вырезки
    device: str
    meta: dict[str, Any] = field(default_factory=dict)
    extra_dim: int = 0                  # числа после эмбеддинга на входе cls (person_inside: person_geo × масштаб)

    def tasks_for_zone(self, zone: str) -> list[str]:
        return [t for t, z in self.zones.items() if z == zone]

    def __call__(self, crops_bgr: Sequence[Any], tasks: Optional[Sequence[str]] = None, *,
                 batch_size: int = 32, extra: Optional[Any] = None) -> dict[str, Any]:
        """{задача: numpy (N,) вероятностей 0..1} для вырезок зоны (BGR, как crop_zone).

        extra — N×extra_dim чисел, приписываемых к эмбеддингу (башня person_inside: геометрия рамки)."""
        import numpy as np
        import torch

        names = list(tasks) if tasks is not None else list(self.tasks)
        missing = [t for t in names if t not in self.tasks]
        if missing:
            raise KeyError(f"в башне {self.name} нет задач {missing} (есть {sorted(self.tasks)})")
        rows = [self.tasks[t] for t in names]
        out: list[Any] = []
        dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            for start in range(0, len(crops_bgr), batch_size):
                chunk = crops_bgr[start:start + batch_size]
                inputs = self.processor(images=[c[:, :, ::-1].copy() for c in chunk], return_tensors="pt")
                px = inputs["pixel_values"].to(self.device, dtype=dtype)
                emb = self.model(pixel_values=px).pooler_output
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                if self.extra_dim:
                    ex = torch.as_tensor(np.asarray(extra[start:start + batch_size], dtype=np.float32), device=emb.device)
                    emb = torch.cat([emb, ex], dim=1)
                logits = emb @ self.weight[rows].T + self.bias[rows]
                out.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(out) if out else np.zeros((0, len(rows)), dtype=np.float32)
        return {t: probs[:, k] for k, t in enumerate(names)}


_CACHE: dict[tuple[str, str], FinetunedTower] = {}


def ft_tower_info(tower: FinetunedTower) -> dict[str, Any]:
    """Сведения для meta сцены: каталог весов, задачи, число блоков, коммит и дата обучения."""
    m = tower.meta
    return {"name": tower.name, "tasks": sorted(tower.tasks), "blocks": m.get("trainable_blocks"),
            "epochs": m.get("epochs"), "commit": m.get("commit"), "date": m.get("date")}


def load_ft_tower(models_dir: Path, rel: str, device: str) -> tuple[Optional[FinetunedTower], Optional[str]]:
    """Башня по каталогу весов rel (относительно models_dir или абсолютному): (башня, None) или (None, ошибка).

    Нет каталога, state.pt/meta.json, базы SigLIP или ключи не сходятся с архитектурой — (None, текст)
    с предупреждением в лог: сцена считается как без флага, задача не падает.
    """
    if not rel:
        return None, None
    path = Path(rel)
    if not path.is_absolute():
        path = Path(models_dir) / path
    key = (str(path.resolve()), device)
    if key in _CACHE:
        return _CACHE[key], None
    try:
        tower = _load(Path(models_dir), path, device)
    except Exception as exc:            # нет файлов, другой формат — работаем как без башни
        msg = f"{path}: {type(exc).__name__}: {exc}"
        LOG.warning("дообученная башня не загружена (%s): сцена считается без неё", msg)
        return None, msg
    _CACHE[key] = tower
    return tower, None


def _load(models_dir: Path, path: Path, device: str) -> FinetunedTower:
    meta = json.loads((path / FT_META).read_text(encoding="utf-8"))   # нет весов — ошибка до импорта torch
    if not (path / FT_STATE).is_file():
        raise FileNotFoundError(f"нет {path / FT_STATE}")

    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    from worker.perception.models import model_path

    state = torch.load(path / FT_STATE, map_location="cpu", weights_only=True)
    base = model_path(models_dir, "zero_shot")
    model = SiglipVisionModel.from_pretrained(base, local_files_only=True)
    vm = model.vision_model
    own = vm.state_dict()
    update: dict[str, Any] = {}
    for name, tensor in state.items():
        if name.startswith("cls."):
            continue
        if not name.startswith(_VM_PREFIX) or name[len(_VM_PREFIX):] not in own:
            raise KeyError(f"лишний ключ {name} в {FT_STATE}")
        sub = name[len(_VM_PREFIX):]
        if tuple(own[sub].shape) != tuple(tensor.shape):
            raise ValueError(f"{name}: форма {tuple(tensor.shape)}, у базы {tuple(own[sub].shape)}")
        update[sub] = tensor.to(own[sub].dtype)
    if not update:
        raise ValueError(f"в {FT_STATE} нет весов башни (ключи {_VM_PREFIX}*)")
    blocks = meta.get("trainable_blocks")
    if blocks is not None:
        layers = len(vm.encoder.layers)
        need = [f"encoder.layers.{i}." for i in range(layers - int(blocks), layers)]
        if not all(any(k.startswith(p) for k in update) for p in need):
            raise ValueError(f"в {FT_STATE} нет весов последних {blocks} блоков")
    vm.load_state_dict({**own, **update})
    weight = state["cls.weight"].float()
    bias = state["cls.bias"].float()
    all_tasks = list(meta.get("all_tasks") or [])
    trained = meta.get("tasks") or [{"name": t, "index": i} for i, t in enumerate(all_tasks)]
    tasks = {str(t["name"]): int(t["index"]) for t in trained}
    zones = {str(t["name"]): str(t.get("zone", "")) for t in trained}
    extra_dim = int(meta.get("extra_dim", 0))
    if not tasks or max(tasks.values()) >= weight.shape[0] or weight.shape[1] != vm.config.hidden_size + extra_dim:
        raise ValueError(f"голова cls {tuple(weight.shape)} не сходится с задачами {tasks}")
    model = model.half() if device == "cuda" else model.float()
    model = model.to(device).eval()
    processor = AutoImageProcessor.from_pretrained(base, local_files_only=True)
    LOG.info("дообученная башня %s: задачи %s, блоков %s (%s)", path, sorted(tasks), blocks, device)
    return FinetunedTower(name=path.name, model=model, processor=processor, weight=weight.to(device),
                          bias=bias.to(device), tasks=tasks, zones=zones, device=device, meta=meta, extra_dim=extra_dim)
