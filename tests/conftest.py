"""Общие фикстуры pytest: корень репозитория в sys.path, синтетическое видео."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """10-секундное тестовое видео 464×848 с кириллической «Х» в имени (как в датасете)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg не установлен")
    out = tmp_path_factory.mktemp("video") / "Х-99.mov"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=464x848:rate=30:duration=10", "-pix_fmt", "yuv420p", str(out)],
        check=True, timeout=60,
    )
    return out


@pytest.fixture(scope="session")
def rules():
    from worker.events import load_rules

    return load_rules(ROOT / "rules.yaml")
