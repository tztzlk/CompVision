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
                   "R14_cash_on_desk_at_entry", "R12_collection_inside", "R13_drink_at_workplace"]
    assert all(r.detector in det.DETECTORS for r in rules)
    # каждый детектор реестра в работе; детекторы позы — в «ЗАПЛАНИРОВАНО» до замера по кэшу
    from worker.events.pose_detectors import POSE_DETECTORS

    used = {r.detector for r in rules}
    assert set(det.DETECTORS) - set(POSE_DETECTORS) <= used <= set(det.DETECTORS)
    assert all(r.source.url.startswith("https://adilet.zan.kz/") for r in rules)


@pytest.mark.parametrize(
    ("name", "rule_id", "t_label"),
    [("X20", "R10_storage_left_open", 5.0), ("X40", "R09_personal_items", 17.0),
     ("X15", "R02_cash_unattended", 15.0), ("X18", "R11_outsider_in_cash_room", 20.0),
     ("X30", "R11_outsider_in_cash_room", 13.0), ("X4", "R14_cash_on_desk_at_entry", 57.0)],
)
def test_positive_fixtures(rules, name, rule_id, t_label):
    events = _events(rules, name)
    assert [e.rule_id for e in events] == [rule_id]
    ev = events[0]
    assert abs(ev.timecode_sec - t_label) <= 3.0          # допуск замера (±3 с)
    assert 1 <= len(ev.evidence_timestamps_sec) <= 3
    assert ev.facts["what_visible"]
    assert len(ev.facts["frames"]) == len(ev.evidence_timestamps_sec)


@pytest.mark.parametrize("name", ["X23", "X3"])
def test_hard_negatives(rules, name):
    """Х-23: кассир у закрытого сейфа; Х-3: клиент входит и садится напротив, на столе пусто."""
    assert _events(rules, name) == []


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


def test_personal_items_description_mentions_only_checked_items():
    """R09 с include_bag=false: сумка не входит в ряд нарушения, значит и в описании её нет (раздел 5 ТЗ)."""
    scene = load_scene(FIXTURES / "scene_X40.json")
    scene = dataclasses.replace(scene, states=[dataclasses.replace(s, bag_on_desk=0.9) for s in scene.states])
    no_bag = det.personal_items(scene, {"include_bag": False})
    assert no_bag and all("сумка" not in ev.facts["what_visible"] for ev in no_bag)
    with_bag = det.personal_items(scene, {"include_bag": True})
    assert with_bag and "сумка" in with_bag[0].facts["what_visible"]


def test_safe_open_from_first_frame_is_its_usual_look(rules):
    """R10: «открыт» на 0,8–0,99 весь ролик (Х-31, Х-43, Х-46) — обычный вид этого сейфа, не событие
    (было: R10 на 0-й секунде)."""
    r10 = next(r for r in rules if r.id == "R10_storage_left_open")
    scene = load_scene(FIXTURES / "scene_X23.json")
    for p in (0.8, 0.99):
        flat = dataclasses.replace(scene, states=tuple(dataclasses.replace(s, safe_open=p) for s in scene.states))
        assert det.storage_left_open(flat, r10.params) == []


def test_safe_open_is_a_change_over_first_seconds(rules):
    """Х-20 на сыром кэше: 0,25 в начале, после открытия 0,6–0,75 и дальше — событие в момент роста;
    короткий рост, пока кассир заслоняет сейф (меньше min_duration_sec), — нет."""
    r10 = next(r for r in rules if r.id == "R10_storage_left_open")
    scene = load_scene(FIXTURES / "scene_X23.json")
    opened = dataclasses.replace(scene, states=tuple(
        dataclasses.replace(s, safe_open=0.25 if s.t < 4.0 else 0.7) for s in scene.states))
    assert [round(e.timecode_sec) for e in det.storage_left_open(opened, r10.params)] == [4]
    blink = dataclasses.replace(scene, states=tuple(
        dataclasses.replace(s, safe_open=0.7 if 8.0 <= s.t < 12.0 else 0.25) for s in scene.states))
    assert det.storage_left_open(blink, r10.params) == []


def test_light_jump_is_not_event(rules):
    """Х-11, Х-12: свет резко меняется — переключение zero-shot-состояния в этот момент не событие."""
    r09 = next(r for r in rules if r.id == "R09_personal_items")
    scene = load_scene(FIXTURES / "scene_X40.json")
    assert det.personal_items(scene, r09.params)                       # пальто на столе с 17 с

    def lit(s):
        return dataclasses.replace(s, extra={**s.extra, "brightness": 0.3 if s.t < 17.0 else 0.6})

    jumped = dataclasses.replace(scene, states=tuple(lit(s) for s in scene.states))
    assert det.personal_items(jumped, r09.params) == []


def test_third_person_without_entry(rules):
    """Х-18 на кэше v1: в комнате трое, а вход не засчитан — R11 всё равно срабатывает."""
    r11 = next(r for r in rules if r.id == "R11_outsider_in_cash_room")
    scene = load_scene(FIXTURES / "scene_X18.json")
    no_entry = tuple(dataclasses.replace(s, person_entered=False,
                                         extra={**s.extra, "visitor_entered_ids": []}) for s in scene.states)
    events = det.outsider_in_cash_room(dataclasses.replace(scene, states=no_entry), r11.params)
    assert [round(e.timecode_sec) for e in events] == [20]


def test_side_by_foot_points():
    """Х-39: кассир спиной к камере накрывает стол, клиент сидит напротив — сторона клиента (< 0);
    по центрам рамок ось вырождалась и клиент попадал на «сторону кассира»."""
    scene = load_scene(FIXTURES / "scene_X3.json")
    scene = dataclasses.replace(scene, zones={**scene.zones, "desk": (0.15, 0.43, 0.71, 0.64),
                                              "workplace": (0.33, 0.40, 0.61, 0.65)})
    client = (0.46, 0.19, 0.58, 0.44)
    assert det._side(scene, client) < 0
    beside_cashier = (0.30, 0.40, 0.45, 0.70)
    assert det._side(scene, beside_cashier) > 0.3


def test_departure_without_valuables_is_not_r02(rules):
    """Х-12: кассир закрыл ноутбук и ушёл — OWLv2 называет закрытый ноутбук «планшетом» почти так же
    уверенно, как ноутбуком; уход без ценностей на столе — не нарушение."""
    scene = load_scene(FIXTURES / "scene_X15.json")
    states = []
    for s in scene.states:
        if s.value("left_items"):
            items = [[16.0, "a laptop", 0.51, [0.48, 0.45, 0.59, 0.49]],
                     [16.0, "a tablet computer", 0.48, [0.48, 0.45, 0.59, 0.49]],
                     [16.0, "a bag", 0.14, [0.10, 0.45, 0.25, 0.54]]]
            s = dataclasses.replace(s, extra={**s.extra, "left_items": items})
        states.append(s)
    events = detect_events(rules, dataclasses.replace(scene, states=tuple(states)))
    assert "R02_cash_unattended" not in [e.rule_id for e in events]
