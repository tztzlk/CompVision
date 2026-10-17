"""scripts/eval_cache.py: замер по кэшу сцен без GPU и pydantic (на фикстурах)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _script():
    spec = importlib.util.spec_from_file_location("eval_cache", ROOT / "scripts" / "eval_cache.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_cache"] = module              # dataclasses ищут модуль в sys.modules
    spec.loader.exec_module(module)
    return module


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("X15", "X18", "X20", "X23", "X40"):
        shutil.copy(FIXTURES / f"scene_{name}.json", cache / f"Х-{name[1:]}.json")
    return cache


def test_summary_on_fixtures(tmp_path, rules):
    ev = _script()
    truth = ev.load_truth(ROOT / "dataset" / "annotations.csv")
    results = ev.evaluate(_cache(tmp_path), truth, rules, 3.0)
    s = ev.summarize(results, rules, 3.0)
    assert [r.outcome for r in results] == ["TP", "TP", "TP", "TN", "TP"]
    assert s["TP"] == 4 and s["TN"] == 1 and s["FP"] == 0
    assert s["timecode_hits"] == s["truth_events"] == 4
    assert s["type_hits"] == s["typed_rows"] == 4


def test_duplicates_counted_once(tmp_path, rules):
    ev = _script()
    cache = _cache(tmp_path)
    shutil.copy(cache / "Х-20.json", cache / "Х-4.json")
    shutil.copy(cache / "Х-20.json", cache / "Х-47.json")       # Х-4 ≡ Х-47 по md5
    truth = ev.load_truth(ROOT / "dataset" / "annotations.csv")
    results = ev.evaluate(cache, truth, rules, 3.0)
    dup = [r for r in results if "X-47" in r.aliases]
    assert len(dup) == 1 and dup[0].truth_timecodes == [57.0, 68.0]
