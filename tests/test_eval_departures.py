"""scripts/eval_departures.py: метрика ухода кассира на синтетических сценах и разметке."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from worker.scene.contract import SceneState, SceneTimeline, save_scene  # noqa: E402

FPS = 3.0
HEADER = "video_id,kind,t_stand,t_leave_workplace,t_exit_room,t_return,room_empty,left_behind,track_switch,note\n"


def _script():
    spec = importlib.util.spec_from_file_location("eval_departures", ROOT / "scripts" / "eval_departures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_departures"] = module              # dataclasses ищут модуль в sys.modules
    spec.loader.exec_module(module)
    return module


def _scene(path: Path, video_id: str, duration: float, *, leave=(), away=(), exit_=(), door_back=(),
           empty=()) -> None:
    """Сцена с шагом 1/3 с (время — float32-подобное, как в кэше: 0.334, 0.667 …).

    leave / exit_ / door_back — секунды флагов; away / empty — интервалы [a, b)."""
    states = []
    n = int(round(duration * FPS))
    for k in range(n):
        t = round(k / FPS + (0.001 if k % 3 == 1 else 0.0), 3)

        def at(points):
            return any(abs(t - p) < 0.17 for p in points)

        def within(spans):
            return any(a - 0.01 <= t < b - 0.01 for a, b in spans)

        states.append(SceneState(
            t=t, frame=f"frames/{k:06d}.jpg", persons_in_room=0 if within(empty) else 1,
            cashier_present=0.0 if within(away) else 1.0,
            extra={"cashier_left_workplace": at(leave), "cashier_away": within(away),
                   "cashier_left_room": at(exit_), "cashier_returned": at(door_back)}))
    save_scene(SceneTimeline(video_id=video_id, duration_sec=duration, fps_sampled=FPS, frame_size=(464, 848),
                             zones={}, states=tuple(states)), path / f"{video_id}.json")


def _labels(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_departure_metric(tmp_path):
    ev = _script()
    cache = tmp_path / "cache"
    # Х-1: уход в 10 с, вышел в 14, в кассе пусто, вернулся через дверь в 25 и на место в 26 — всё найдено
    _scene(cache, "Х-1", 30, leave=[10.0], away=[(11.0, 26.0)], exit_=[14.0], door_back=[25.0], empty=[(14.0, 25.0)])
    # X-2 (латинская X в кэше): уход отмечен на 4.7 с позже — эпизод замечен, таймкод мимо
    _scene(cache, "X-2", 30, leave=[12.67], away=[(13.0, 30.0)])
    # Х-3: уход пропущен; ложный уход на эпизоде not_departure (сейф)
    _scene(cache, "Х-3", 30, leave=[21.0], away=[(21.5, 26.0)])
    # Х-4: kind=none, а модель отметила уход — ложный на ролике без ухода
    _scene(cache, "Х-4", 20, leave=[5.0], away=[(5.5, 20.0)])
    labels = _labels(tmp_path, "\n".join([
        "Х-1,departure,9.33,10.67,14.0,26.33,да,наличные,нет,",
        "Х-2,departure,7.0,8.0,?,,?,ничего,да,",
        "Х-3,departure,4.0,5.0,,15.0,нет,сумка,нет,",
        "Х-3,not_departure,20.0,20.67,,24.0,,,,сейф",
        "Х-4,none,,,,,,,,",
    ]) + "\n")
    res = ev.evaluate(cache, ev.load_labels(labels), 3.0)
    s = res["summary"]
    assert s["videos"] == 4 and s["episodes"] == {"departure": 3, "not_departure": 1}
    leave = s["leave"]
    assert (leave["truth"], leave["found"], leave["late"], leave["missed"]) == (3, 1, 1, 1)
    assert (leave["false_not_departure"], leave["false_no_departure"], leave["false_other"]) == (1, 1, 0)
    assert leave["timecode_ok"] == 1 and leave["timecode_n"] == 2
    ex = s["exit"]
    assert (ex["truth"], ex["found"], ex["false"]) == (1, 1, 0)
    ret = s["return"]
    assert ret["found"] == 1 and ret["no_return_ok"] == 1 and ret["no_departure"] == 1
    assert s["door_return"]["found"] == 1
    assert s["room_empty"] == {"n": 2, "ok": 2}           # Х-1 «да» и Х-3 «нет» (уход не найден, людей 1)
    assert s["track_switch"] == {"n": 3, "yes": 1}


def test_question_marks_not_counted(tmp_path):
    ev = _script()
    cache = tmp_path / "cache"
    _scene(cache, "Х-5", 20, leave=[6.0], away=[(6.5, 20.0)], exit_=[9.0])
    labels = _labels(tmp_path, "Х-5,departure,?,6.33,?,,?,?,?,\n")
    s = ev.evaluate(cache, ev.load_labels(labels), 3.0)["summary"]
    assert s["leave"]["found"] == 1
    assert s["exit"]["truth"] == 0 and s["exit"]["false"] == 0     # выход внутри эпизода с «?» — не ложный
    assert s["room_empty"]["n"] == 0 and s["track_switch"]["n"] == 0


def test_match_is_one_to_one():
    ev = _script()
    assert ev.match([(0, 10.0), (1, 11.0)], [10.9], 3.0) == {1: 0}
    assert ev.match([(0, 10.0)], [13.5], 3.0) == {}
    assert ev.ms(0.334) == 334 and ev.ms(1.0009) == 1001
