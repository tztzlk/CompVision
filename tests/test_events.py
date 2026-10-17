"""worker.events: детекторы на фикстурах SceneTimeline (раздел 3 задания) и общая механика."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from worker.events import RulesError, detect_events, load_rules
from worker.events import detectors as det
from worker.scene.contract import load_scene

FIXTURES = Path(__file__).parent / "fixtures"


def _events(rules, name: str):
    return detect_events(rules, load_scene(FIXTURES / f"scene_{name}.json"))


def test_rules_yaml_loads_all_enabled(rules):
    ids = [r.id for r in rules]
    assert ids == ["R01_left_during_service", "R02_cash_unattended", "R06_phone_at_workplace",
                   "R09_personal_items", "R10_storage_left_open", "R11_outsider_in_cash_room",
                   "R12_collection_inside", "R13_drink_at_workplace"]
    assert all(r.detector in det.DETECTORS for r in rules)
    assert {r.detector for r in rules} == set(det.DETECTORS)   # каждый детектор реестра в работе
    assert all(r.source.url.startswith("https://adilet.zan.kz/") for r in rules)


@pytest.mark.parametrize(
    ("name", "rule_id", "t_label"),
    [("X20", "R10_storage_left_open", 5.0), ("X40", "R09_personal_items", 17.0),
     ("X15", "R02_cash_unattended", 15.0), ("X18", "R11_outsider_in_cash_room", 20.0)],
)
def test_positive_fixtures(rules, name, rule_id, t_label):
    events = _events(rules, name)
    assert [e.rule_id for e in events] == [rule_id]
    ev = events[0]
    assert abs(ev.timecode_sec - t_label) <= 1.0
    assert 1 <= len(ev.evidence_timestamps_sec) <= 3
    assert ev.facts["what_visible"]
    assert len(ev.facts["frames"]) == len(ev.evidence_timestamps_sec)


def test_hard_negative_x23(rules):
    assert _events(rules, "X23") == []


def test_result_is_valid(rules, tmp_path, synthetic_video):
    from shared.schemas import AnalysisResult, Verdict
    from worker.verdict import build_result

    res = build_result("job", _events(rules, "X20"), rules, tmp_path, synthetic_video)
    assert res.verdict is Verdict.VIOLATION
    assert "дверца сейфа открыта" in res.violations[0].description
    AnalysisResult.model_validate(res.model_dump())


def test_null_is_no_data():
    vals = [None, 0.9, None, 0.9, 0.9, None, 0.1]
    assert det.hysteresis(vals, 0.6, 0.4) == [det.Interval(1, 4)]
    assert det.hysteresis([None] * 5, 0.6, 0.4) == []
    assert det.smooth([None, 0.9, None], 3) == [None, 0.9, None]


def test_hysteresis_and_merge():
    vals = [0.0, 0.7, 0.5, 0.5, 0.3, 0.0, 0.8, 0.0]
    ivs = det.hysteresis(vals, 0.6, 0.4)
    assert ivs == [det.Interval(1, 3), det.Interval(6, 6)]
    times = [i / 3 for i in range(len(vals))]
    assert det.merge_intervals(ivs, times, 2.0) == [det.Interval(1, 6)]


def test_unknown_detector_rejected(tmp_path):
    bad = tmp_path / "rules.yaml"
    bad.write_text(
        "rules:\n  - id: X\n    title: t\n    detector: nope\n    params: {}\n"
        "    source: {doc: d, url: u, clause: c}\n    template: t\n",
        encoding="utf-8",
    )
    with pytest.raises(RulesError, match="неизвестный детектор"):
        load_rules(bad)


def test_broken_detector_does_not_fail_job(rules, monkeypatch):
    def boom(scene, params):
        raise RuntimeError("сломался")

    monkeypatch.setitem(det.DETECTORS, "storage_left_open", boom)
    events = _events(rules, "X20")
    assert events == []          # R10 упал, остальные отработали и ничего не нашли
    rules2 = [dataclasses.replace(r) for r in rules]
    assert _events(rules2, "X40")[0].rule_id == "R09_personal_items"
