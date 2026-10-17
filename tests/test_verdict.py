"""worker.verdict: сборка AnalysisResult из событий."""

from __future__ import annotations

import base64

from shared.schemas import AnalysisResult, Verdict
from worker.events import Event
from worker.verdict import MAX_VIOLATIONS, build_result


def _event(rule_id: str, t: float, conf: float = 0.9, **facts) -> Event:
    return Event(rule_id=rule_id, timecode_sec=t, confidence=conf,
                 evidence_timestamps_sec=[t, t + 1.0], facts={"what_visible": "тест", **facts})


def test_no_events_gives_no_violation(rules, tmp_path):
    res = build_result("job", [], rules, tmp_path, None)
    assert res.verdict is Verdict.NO_VIOLATION and res.violations == []
    AnalysisResult.model_validate(res.model_dump())


def test_frames_cut_from_video_with_cyrillic_name(rules, tmp_path, synthetic_video):
    res = build_result("job", [_event("R02", 5.0)], rules, tmp_path, synthetic_video)
    assert res.verdict is Verdict.VIOLATION
    v = res.violations[0]
    assert v.type == next(r.title for r in rules if r.id == "R02")
    assert v.timecode_sec == 5.0
    assert [f.timestamp_sec for f in v.evidence_frames] == [5.0, 6.0]
    assert base64.b64decode(v.evidence_frames[0].image_base64).startswith(b"\xff\xd8")
    assert v.rule is not None and v.rule.url.startswith("https://adilet.zan.kz/")
    assert "п. 20" in v.description and "{" not in v.description
    AnalysisResult.model_validate(res.model_dump())


def test_prepared_frame_is_used(rules, tmp_path, synthetic_video):
    from worker.verdict import extract_frame

    extract_frame(synthetic_video, 2.0, tmp_path / "frames" / "000006.jpg")
    ev = Event(rule_id="R01", timecode_sec=2.0, confidence=0.9, evidence_timestamps_sec=[2.0],
               facts={"what_visible": "пальто", "frames": ["frames/000006.jpg"]})
    res = build_result("job", [ev], rules, tmp_path, None)   # без видео: только готовый кадр
    assert len(res.violations) == 1 and len(res.violations[0].evidence_frames) == 1


def test_low_confidence_and_unknown_rule_dropped(rules, tmp_path, synthetic_video):
    events = [_event("R01", 3.0, conf=0.1), _event("NOPE", 4.0)]
    res = build_result("job", events, rules, tmp_path, synthetic_video)
    assert res.verdict is Verdict.NO_VIOLATION


def test_no_frames_drops_violation(rules, tmp_path):
    res = build_result("job", [_event("R01", 3.0)], rules, tmp_path, None)
    assert res.verdict is Verdict.NO_VIOLATION


def test_limit_and_time_order(rules, tmp_path, synthetic_video):
    events = [_event("R04", float(t), conf=0.5 + t / 100) for t in range(8)]
    res = build_result("job", events, rules, tmp_path, synthetic_video)
    times = [v.timecode_sec for v in res.violations]
    assert len(times) == MAX_VIOLATIONS
    assert times == sorted(times) and min(times) >= 3.0   # остались самые уверенные
