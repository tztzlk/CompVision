"""Локальные модели восприятия: RT-DETR (COCO), OWLv2 (open-vocabulary), SigLIP (zero-shot).

torch и transformers импортируются лениво, внутри конструкторов: модуль можно импортировать
на машине без них (тесты, events у человека B). Веса — в <models_dir>/hf/<каталог>, их кладёт
scripts/download_models.py; загрузка только с диска (local_files_only), без сети.

Лицензии (проверено на страницах моделей 17.10.2026): все три модели — Apache-2.0.
Выходы моделей разбираются вручную (logits/pred_boxes), без post_process_* — так код меньше
зависит от версии transformers и не путается в размерах кадра с паддингом.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Optional, Sequence

from worker.perception import Detection
from worker.perception.geometry import clip

__all__ = [
    "ModelRef",
    "MODELS",
    "ModelNotFoundError",
    "model_path",
    "resolve_device",
    "CocoDetector",
    "OpenVocabDetector",
    "ZeroShotClassifier",
    "ModelBundle",
    "load_models",
]

LOG = logging.getLogger("worker.perception")


@dataclass(frozen=True)
class ModelRef:
    key: str
    repo: str           # репозиторий на Hugging Face
    revision: str       # зафиксированный коммит (как в scripts/download_models.py)
    local_dir: str      # каталог в <models_dir>/hf/
    license: str = "Apache-2.0"


MODELS: dict[str, ModelRef] = {
    "person_detector": ModelRef(
        "person_detector", "PekingU/rtdetr_r18vd_coco_o365",
        "386c43855d01bcd966e548f3296fbd2cd2a2a40b", "rtdetr_r18vd_coco_o365"),
    "open_vocab": ModelRef(
        "open_vocab", "google/owlv2-base-patch16-ensemble",
        "cfd3195ba4ea9592eec887ded089f4c08eff231d", "owlv2-base-patch16-ensemble"),
    "zero_shot": ModelRef(
        "zero_shot", "google/siglip-base-patch16-224",
        "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", "siglip-base-patch16-224"),
}


class ModelNotFoundError(FileNotFoundError):
    """Нет локальных весов модели."""


def model_path(models_dir: Path, key: str) -> Path:
    """Каталог весов модели; ошибка с подсказкой, если веса не скачаны."""
    ref = MODELS[key]
    path = Path(models_dir) / "hf" / ref.local_dir
    if not (path / "config.json").is_file():
        raise ModelNotFoundError(
            f"нет весов {ref.repo} в {path}: один раз запустите python scripts/download_models.py")
    return path


def resolve_device(device: str) -> str:
    """cuda, если запрошена и доступна; иначе cpu (с предупреждением)."""
    import torch

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        LOG.warning("CUDA недоступна, инференс пойдёт на CPU (медленно)")
    return "cpu"


def _to_rgb(image_bgr: Any) -> Any:
    return image_bgr[:, :, ::-1].copy()


class _TorchModel:
    def __init__(self, device: str) -> None:
        import torch

        self._torch = torch
        self.device = device

    def _autocast(self) -> Any:
        if self.device == "cuda":
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return contextlib.nullcontext()

    def _to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}


class CocoDetector(_TorchModel):
    """RT-DETR R18 (COCO, 80 классов): люди и предметы (телефон, стакан, сумка, ноутбук...)."""

    def __init__(self, path: Path, device: str) -> None:
        super().__init__(device)
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self.processor = RTDetrImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = RTDetrForObjectDetection.from_pretrained(path, local_files_only=True).to(device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}

    def __call__(
        self,
        images_bgr: Sequence[Any],
        *,
        score_threshold: float = 0.3,
        labels: Optional[Collection[str]] = None,
    ) -> list[list[Detection]]:
        if not images_bgr:
            return []
        torch = self._torch
        inputs = self.processor(images=[_to_rgb(im) for im in images_bgr], return_tensors="pt")
        with torch.inference_mode(), self._autocast():
            outputs = self.model(**self._to_device(inputs))
        probs = outputs.logits.float().sigmoid()      # (B, Q, C): RT-DETR обучен с focal loss
        scores, classes = probs.max(dim=-1)
        boxes = outputs.pred_boxes.float()             # (B, Q, 4): cx, cy, w, h в долях кадра
        wanted = set(labels) if labels else None
        result: list[list[Detection]] = []
        for b in range(len(images_bgr)):
            dets: list[Detection] = []
            for q in (scores[b] >= score_threshold).nonzero(as_tuple=True)[0].tolist():
                class_id = int(classes[b, q])
                label = self.id2label.get(class_id, str(class_id))
                if wanted is not None and label not in wanted:
                    continue
                cx, cy, w, h = boxes[b, q].tolist()
                dets.append(Detection(label=label, score=round(float(scores[b, q]), 4),
                                      bbox_xyxy=clip((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))))
            result.append(dets)
        return result


class OpenVocabDetector(_TorchModel):
    """OWLv2: поиск предметов по текстовым запросам (зоны: стол, сейф, дверь, ноутбук)."""

    def __init__(self, path: Path, device: str) -> None:
        super().__init__(device)
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(path, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(path, local_files_only=True).to(device).eval()

    def __call__(
        self,
        images_bgr: Sequence[Any],
        queries: Sequence[str],
        *,
        score_threshold: float = 0.1,
        top_k: int = 3,
    ) -> list[list[Detection]]:
        torch = self._torch
        texts = list(queries)
        result: list[list[Detection]] = []
        for image in images_bgr:                       # по одному кадру: модель тяжёлая, VRAM 4–8 ГБ
            h, w = image.shape[:2]
            side = float(max(h, w))
            inputs = self.processor(text=[texts], images=[_to_rgb(image)], return_tensors="pt")
            with torch.inference_mode(), self._autocast():
                outputs = self.model(**self._to_device(inputs))
            scores = outputs.logits[0].float().sigmoid()   # (патчи, запросы)
            boxes = outputs.pred_boxes[0].float()            # (патчи, 4): cx, cy, w, h в долях квадрата
            # OWLv2 дополняет кадр до квадрата серым снизу и справа: перевод в доли исходного кадра.
            sx, sy = side / w, side / h
            dets: list[Detection] = []
            for qi, query in enumerate(texts):
                values, indices = scores[:, qi].topk(min(top_k, scores.shape[0]))
                for score, idx in zip(values.tolist(), indices.tolist()):
                    if score < score_threshold:
                        continue
                    cx, cy, bw, bh = boxes[idx].tolist()
                    dets.append(Detection(label=query, score=round(score, 4), bbox_xyxy=clip((
                        (cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy))))
            result.append(dets)
        return result


class ZeroShotClassifier(_TorchModel):
    """SigLIP: zero-shot вероятность состояния по вырезке зоны (позитивные против негативных подписей)."""

    def __init__(self, path: Path, device: str) -> None:
        super().__init__(device)
        from transformers import AutoProcessor, SiglipModel

        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = SiglipModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self._text_cache: dict[tuple[str, ...], Any] = {}

    def _text_embeddings(self, prompts: tuple[str, ...]) -> Any:
        cached = self._text_cache.get(prompts)
        if cached is not None:
            return cached
        torch = self._torch
        # SigLIP обучали с padding="max_length" (64 токена) — иначе эмбеддинги текста искажаются.
        inputs = self.processor(text=list(prompts), padding="max_length", return_tensors="pt")
        with torch.inference_mode():
            emb = self.model.get_text_features(input_ids=inputs["input_ids"].to(self.device))
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        self._text_cache[prompts] = emb
        return emb

    def __call__(
        self,
        crops_bgr: Sequence[Any],
        prompt_sets: Mapping[str, tuple[Sequence[str], Sequence[str]]],
        *,
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """Для каждой вырезки — {состояние: вероятность} по его (позитивным, негативным) подписям."""
        if not crops_bgr or not prompt_sets:
            return [{} for _ in crops_bgr]
        torch = self._torch
        prompts = tuple(dict.fromkeys(p for pos, neg in prompt_sets.values() for p in (*pos, *neg)))
        index = {p: i for i, p in enumerate(prompts)}
        groups = {name: ([index[p] for p in (*pos, *neg)], len(pos)) for name, (pos, neg) in prompt_sets.items()}
        text = self._text_embeddings(prompts)
        result: list[dict[str, float]] = []
        # inference_mode на всё вычисление: logit_scale — параметр с requires_grad, без этого
        # torch строит граф автоградиента на каждой вырезке (лишняя память и предупреждения).
        with torch.inference_mode():
            scale = self.model.logit_scale.exp().float()
            for start in range(0, len(crops_bgr), batch_size):
                chunk = crops_bgr[start:start + batch_size]
                inputs = self.processor(images=[_to_rgb(c) for c in chunk], return_tensors="pt")
                with self._autocast():
                    image = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device))
                image = torch.nn.functional.normalize(image.float(), dim=-1)
                logits = scale * image @ text.T        # logit_bias SigLIP общий для всех подписей и в softmax сокращается
                for row in logits:
                    probs: dict[str, float] = {}
                    for name, (idx, n_pos) in groups.items():
                        soft = torch.softmax(row[idx], dim=-1)
                        probs[name] = round(float(soft[:n_pos].sum()), 4)
                    result.append(probs)
        return result


@dataclass(frozen=True)
class ModelBundle:
    detector: CocoDetector
    open_vocab: OpenVocabDetector
    zero_shot: ZeroShotClassifier
    device: str
    info: Mapping[str, str]          # {ключ: "repo@ревизия"} — для meta в SceneTimeline


_CACHE: dict[tuple[str, str], ModelBundle] = {}


def load_models(models_dir: Path, device: str) -> ModelBundle:
    """Загрузить все модели один раз на процесс (повторные вызовы берут их из кэша)."""
    key = (str(Path(models_dir).resolve()), device)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    resolved = resolve_device(device)
    started = time.monotonic()
    bundle = ModelBundle(
        detector=CocoDetector(model_path(models_dir, "person_detector"), resolved),
        open_vocab=OpenVocabDetector(model_path(models_dir, "open_vocab"), resolved),
        zero_shot=ZeroShotClassifier(model_path(models_dir, "zero_shot"), resolved),
        device=resolved,
        info={k: f"{ref.repo}@{ref.revision[:7]}" for k, ref in MODELS.items()},
    )
    LOG.info("модели загружены за %.1f с (device=%s)", time.monotonic() - started, resolved)
    _CACHE[key] = bundle
    return bundle
