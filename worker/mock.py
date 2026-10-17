"""TODO(МОК): удалить этот модуль целиком, когда worker.runner.process_video получит настоящий пайплайн.

МОК-режим (контракт, раздел 8): перцепции нет. Для видео:
    ffprobe: длительность -> ffmpeg: один кадр в середине -> work_dir/frames/mock.jpg
    -> AnalysisResult(done, violation, одно тестовое нарушение по первому правилу rules.yaml).
Вызывается только из worker.runner.analyze (единственное место выбора мок/не мок).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from shared.schemas import AnalysisResult, JobStatus, Verdict, Violation
from worker.events import Rule
from worker.verdict import encode_frame, render_description, rule_ref

__all__ = ["probe_duration", "extract_frame", "mock_analyze"]

FFPROBE_TIMEOUT_SEC = 30
FFMPEG_TIMEOUT_SEC = 60
MOCK_FRAME_NAME = "mock.jpg"
MOCK_TYPE = "ТЕСТ: заглушка worker"
MOCK_WHAT_VISIBLE = "тестовый кадр из середины видео (МОК-режим, анализ не выполнялся)"


def _run(cmd: list[str], timeout: float) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cmd[0]}: превышен таймаут {timeout} с") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-500:]
        raise RuntimeError(f"{cmd[0]} завершился с кодом {proc.returncode}: {tail}")
    return proc.stdout


def probe_duration(video: Path) -> float:
    """Длительность видео в секундах (ffprobe)."""
    out = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         "-select_streams", "v:0", str(video)],
        FFPROBE_TIMEOUT_SEC,
    )
    info = json.loads(out or "{}")
    if not info.get("streams"):
        raise RuntimeError("в файле нет видеопотока")
    for raw in (info.get("format", {}).get("duration"), info["streams"][0].get("duration")):
        try:
            duration = float(raw)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    raise RuntimeError("не удалось определить длительность видео")


def extract_frame(video: Path, timestamp_sec: float, out_path: Path) -> None:
    """Один JPEG-кадр на timestamp_sec (ffmpeg)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{timestamp_sec:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "3", "-f", "image2", str(out_path)],
        FFMPEG_TIMEOUT_SEC,
    )
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg не создал кадр")


def mock_analyze(job_id: str, video: Path, work_dir: Path, rule: Rule) -> AnalysisResult:
    """TODO(МОК): одно тестовое «нарушение» с кадром из середины видео."""
    if not video.is_file():
        raise RuntimeError(f"видео не найдено: {video}")
    duration = probe_duration(video)
    mid = round(duration / 2, 1)
    frame_path = work_dir / "frames" / MOCK_FRAME_NAME
    extract_frame(video, mid, frame_path)
    violation = Violation(
        type=MOCK_TYPE,
        timecode_sec=mid,
        description=render_description(rule, mid, {"what_visible": MOCK_WHAT_VISIBLE}),
        evidence_frames=[encode_frame(frame_path, mid)],
        rule=rule_ref(rule),
    )
    return AnalysisResult(
        job_id=job_id, status=JobStatus.DONE, verdict=Verdict.VIOLATION, violations=[violation]
    )
