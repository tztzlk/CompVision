"""scripts/evaluate.py --from-cache: правила и вердикт по готовым сценам, без моделей."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("script_evaluate", ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses ищут модуль в sys.modules при разборе аннотаций
    spec.loader.exec_module(module)
    return module


def _video(path: Path, seconds: int) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size=464x848:rate=5:duration={seconds}", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=60,
    )


def test_from_cache(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg не установлен")
    videos, cache = tmp_path / "dataset", tmp_path / "cache"
    videos.mkdir()
    cache.mkdir()
    # Х-20: открытый сейф на 5-й с (нарушение); Х-23: без нарушений; Х-30: кэша нет.
    for name in ("Х-20", "Х-23", "Х-30"):
        _video(videos / f"{name}.mov", 21)
    shutil.copy(FIXTURES / "scene_X20.json", cache / "Х-20.json")
    shutil.copy(FIXTURES / "scene_X23.json", cache / "Х-23.json")
    ann = tmp_path / "annotations.csv"
    ann.write_text("video_id,t_label_sec\nX-20,5\nХ-23,\nX-30,13\n", encoding="utf-8")
    out = tmp_path / "report.json"

    code = _load_evaluate().main([
        "--dataset", str(videos), "--annotations", str(ann), "--rules", str(ROOT / "rules.yaml"),
        "--from-cache", str(cache), "--json", str(out),
    ])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    s = report["summary"]
    assert (s["TP"], s["FN"], s["FP"], s["TN"], s["ERR"]) == (1, 0, 0, 1, 1)
    assert s["matched_events"] == 1
    by_id = {v["video_id"]: v for v in report["videos"]}
    assert by_id["X-20"]["pred_types"] == ["Сейф с наличными оставлен открытым"]
    assert "нет кэша" in by_id["X-30"]["error"]
    rules = {r["rule_id"]: r for r in report["rules"]}
    r10 = rules["R10_storage_left_open"]
    assert (r10["videos_violation"], r10["videos_clean"], r10["events_hit"], r10["early"]) == (1, 0, 1, 0)
    assert rules["R11_outsider_in_cash_room"]["fired"] == 0     # правило без срабатываний тоже в отчёте


def test_from_cache_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _load_evaluate().main(["--from-cache", str(tmp_path / "нет")])
