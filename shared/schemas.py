"""Pydantic-модели ответа API строго по разделу 10 ТЗ.

Единственный источник правды по формату для backend (отдаёт JSON жюри) и worker
(формирует результат). Поля и значения перечислений не менять: жюри проверяет
результаты автоматически по единой схеме.

Пример результата из ТЗ:
    {
      "job_id": "3f2a9c1e",
      "status": "done",
      "verdict": "violation",
      "violations": [
        {
          "type": "Название типа нарушения",
          "timecode_sec": 44.2,
          "description": "Что видно на видео и почему это нарушение",
          "evidence_frames": [{"timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..."}]
        }
      ]
    }

Импорт в контейнерах: см. shared/__init__.py (shared/ копируется в образы при сборке).

Модели ответа допускают лишние поля при разборе (extra="ignore"): строгость нужна
только при валидации входа (rules.yaml в worker), а не при чтении сохранённых результатов.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "JobStatus",
    "Verdict",
    "EvidenceFrame",
    "RuleRef",
    "Violation",
    "AnalysisResult",
    "AnalyzeResponse",
]


class _Response(BaseModel):
    """База моделей ответа: лишние поля при разборе отбрасываются (extra="ignore")."""

    model_config = ConfigDict(extra="ignore")


class JobStatus(str, Enum):
    """Статус обработки задачи (GET /api/v1/jobs/{job_id})."""

    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Verdict(str, Enum):
    """Вердикт по всему видео (раздел 4, п. 3 ТЗ)."""

    NO_VIOLATION = "no_violation"
    VIOLATION = "violation"


class EvidenceFrame(_Response):
    """Кадр-доказательство: JPEG в base64 и его таймкод."""

    timestamp_sec: float = Field(ge=0, description="Секунды от начала видео.")
    image_base64: str = Field(min_length=1, description="JPEG, закодированный в base64.")


class RuleRef(_Response):
    """Правило и нормативный источник (раздел 4, п. 7 ТЗ). Необязательное дополнение
    к формату раздела 10: тот же текст всегда дублируется в Violation.description."""

    id: str = Field(min_length=1, description="Идентификатор правила из rules.yaml.")
    doc: str = Field(min_length=1, description="Название нормативного документа.")
    url: str = Field(min_length=1, description="Ссылка на документ.")
    clause: str = Field(min_length=1, description="Пункт/раздел документа.")


class Violation(_Response):
    """Одно возможное нарушение в видео."""

    type: str = Field(min_length=1, description="Тип нарушения.")
    timecode_sec: float = Field(ge=0, description="Таймкод события в секундах от начала видео.")
    description: str = Field(
        min_length=1, description="Что видно на видео и почему это может быть нарушением."
    )
    evidence_frames: list[EvidenceFrame] = Field(
        min_length=1, max_length=3, description="1–3 кадра-доказательства."
    )
    rule: Optional[RuleRef] = Field(
        default=None, description="Правило и источник; при None не сериализуется (exclude_none)."
    )


class AnalysisResult(_Response):
    """Ответ GET /api/v1/jobs/{job_id}.

    Пока status != done, verdict = None и violations пуст.
    При status == done verdict обязателен и согласован со списком нарушений.
    """

    job_id: str
    status: JobStatus
    verdict: Optional[Verdict] = None
    violations: list[Violation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnalysisResult":
        if self.status is JobStatus.DONE:
            if self.verdict is None:
                raise ValueError("verdict обязателен при status=done")
            has_violations = bool(self.violations)
            if has_violations != (self.verdict is Verdict.VIOLATION):
                raise ValueError("verdict=violation тогда и только тогда, когда violations не пуст")
        elif self.verdict is not None or self.violations:
            raise ValueError("verdict и violations задаются только при status=done")
        return self


class AnalyzeResponse(_Response):
    """Ответ POST /api/v1/analyze: идентификатор созданной задачи."""

    job_id: str
    status: JobStatus = JobStatus.PROCESSING
