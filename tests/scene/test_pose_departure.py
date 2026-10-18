"""Поза кассира в уходе с места (worker.scene.pose_departure, SceneConfig.pose_departure) — синтетика, без моделей."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from worker.perception.tracking import Track, TrackPoint
from worker.scene.builder import FrameRef, SceneConfig, _departures, assemble_timeline
from worker.scene.contract import scene_to_dict
from worker.scene.pose_departure import (anchor_point, feet_behind_glass, is_standing, legs_visible, match_pose,
                                         pose_by_frame, stood_up_at)
from worker.scene.raw_cache import load_raw, rebuild_scene, save_raw

ROOT = Path(__file__).resolve().parents[2]
FPS = 3.0
N = 30
TIMES = [round(k / FPS, 3) for k in range(N)]
DESK = (0.10, 0.45, 0.70, 0.70)
DOOR = (0.00, 0.05, 0.25, 0.50)
SEATED = (0.30, 0.35, 0.55, 0.80)
GLASS = (0.33, 0.25, 0.52, 0.47)            # за стеклом над местом: низ рамки у верха стола
AT_DOOR = (0.02, 0.20, 0.22, 0.62)
WORKPLACE = (0.20, 0.30, 0.65, 0.66)        # центр между сидячей рамкой и рамкой за стеклом (d 0,21 у обеих)


def kp(*, hip=(0.42, 0.55), knee=(0.42, 0.65), ankle=(0.42, 0.75), knee_score=0.9, ankle_score=0.9,
       hip_score=0.9) -> list[list[float]]:
    """17 точек COCO: верх тела уверенно, таз/колени/лодыжки — как заданы."""
    pts = [[0.42, 0.3, 0.9] for _ in range(11)]
    pts += [[hip[0] - 0.02, hip[1], hip_score], [hip[0] + 0.02, hip[1], hip_score]]
    pts += [[knee[0] - 0.02, knee[1], knee_score], [knee[0] + 0.02, knee[1], knee_score]]
    pts += [[ankle[0] - 0.02, ankle[1], ankle_score], [ankle[0] + 0.02, ankle[1], ankle_score]]
    return pts


SEATED_KP = kp(hip=(0.42, 0.60), knee=(0.52, 0.62), ankle=(0.52, 0.75), ankle_score=0.03)   # ноги за столом
GLASS_KP = kp(hip=(0.42, 0.36), knee=(0.42, 0.41), ankle=(0.42, 0.46), knee_score=0.75, ankle_score=0.6)
ROOM_STAND_KP = kp(hip=(0.42, 0.60), knee=(0.42, 0.75), ankle=(0.42, 0.90))                  # перед столом


def test_legs_visible() -> None:
    assert legs_visible(GLASS_KP, 0.4)
    assert not legs_visible(SEATED_KP, 0.4)                      # лодыжки закрыты столом
    assert not legs_visible(kp(knee_score=0.1), 0.4)             # нет колен
    assert not legs_visible(None, 0.4)
    assert not legs_visible(GLASS_KP, 0.7)                       # порог выше уверенности лодыжек


def test_feet_behind_glass() -> None:
    top = DESK[1]
    assert feet_behind_glass(GLASS_KP, top, 0.4, 0.03)            # во весь рост, ступни у верха стола
    assert not feet_behind_glass(SEATED_KP, top, 0.4, 0.03)       # сидит: ног не видно
    assert not feet_behind_glass(ROOM_STAND_KP, top, 0.4, 0.03)   # стоит в комнате: ступни ниже стола
    assert not feet_behind_glass(GLASS_KP, None, 0.4, 0.03)       # стола нет — признак молчит


def test_anchor_and_standing() -> None:
    assert anchor_point(ROOM_STAND_KP, 0.4) == (0.42, 0.60)
    assert anchor_point(kp(hip_score=0.1), 0.4) is None
    assert is_standing(ROOM_STAND_KP, 0.4, 0.8) is True
    assert is_standing(SEATED_KP, 0.4, 0.8) is False             # бедро почти горизонтально
    assert is_standing(kp(knee_score=0.0), 0.4, 0.8) is None
    times = [0.0, 0.33, 0.67, 1.0, 1.33]
    assert stood_up_at([False, False, True, True, True], times, 4, 3.0) == 2
    assert stood_up_at([True, True, True, True, True], times, 4, 3.0) == 4    # сидя не видели — как было
    assert stood_up_at([False, False, False, False, False], times, 4, 3.0) == 4


def test_match_pose_by_iou() -> None:
    index = pose_by_frame([[3, list(SEATED), SEATED_KP], [3, list(GLASS), GLASS_KP]])
    assert match_pose(index, 3, SEATED) is SEATED_KP
    assert match_pose(index, 3, (0.31, 0.35, 0.55, 0.80)) is SEATED_KP        # округление рамки
    assert match_pose(index, 4, SEATED) is None
    assert match_pose(index, 3, AT_DOOR) is None
    assert match_pose(index, 3, None) is None


def _arrival():
    """Х-31/Х-32: до входа кассир за стеклом над местом (score высокий), идёт к двери, входит и садится."""
    boxes, scores, pose = [], [], []
    for k in range(N):
        if k < 9:
            boxes.append(GLASS), scores.append(0.9), pose.append(GLASS_KP)
        elif k < 21:
            boxes.append(AT_DOOR), scores.append(0.9), pose.append(None)
        else:
            boxes.append(SEATED), scores.append(0.95), pose.append(SEATED_KP)
    return boxes, scores, pose


def test_glass_point_is_not_home() -> None:
    boxes, scores, pose = _arrival()
    cfg = SceneConfig(fps=FPS)
    base = _departures(boxes, scores, TIMES, WORKPLACE, None, cfg)
    assert len(base) == 1 and base[0].left_pos == 8                      # без позы — ложный уход
    on = dataclasses.replace(cfg, pose_departure=True)
    assert _departures(boxes, scores, TIMES, WORKPLACE, None, on, pose=pose, desk=DESK) == []
    # без точек на кадрах — как без позы
    assert _departures(boxes, scores, TIMES, WORKPLACE, None, on, pose=[None] * N, desk=DESK) == base
    # подфлаг выключен — как без позы
    off_glass = dataclasses.replace(on, pose_dep_glass=False)
    assert _departures(boxes, scores, TIMES, WORKPLACE, None, off_glass, pose=pose, desk=DESK) == base


def test_seated_cashier_stays_home_with_pose() -> None:
    """Сидящий за столом (ноги закрыты) с позой остаётся «у места»: настоящий уход находится так же."""
    boxes = [SEATED if k < 12 or k >= 27 else AT_DOOR for k in range(N)]
    pose = [SEATED_KP if b == SEATED else None for b in boxes]
    cfg = SceneConfig(fps=FPS)
    base = _departures(boxes, [0.95] * N, TIMES, WORKPLACE, None, cfg)
    on = _departures(boxes, [0.95] * N, TIMES, WORKPLACE, None, dataclasses.replace(cfg, pose_departure=True),
                     pose=pose, desk=DESK)
    assert base == on and len(on) == 1 and on[0].left_pos == 11


def _tracks():
    boxes, scores, _ = _arrival()
    return [Track(1, [TrackPoint(k, TIMES[k], boxes[k], scores[k]) for k in range(N)])]


def _person_pose():
    boxes, _, pose = _arrival()
    return [[k, list(boxes[k]), pose[k]] for k in range(N) if pose[k] is not None]


def _timeline(config, person_pose=None):
    frames = [FrameRef(k, TIMES[k], f"frames/{k:06d}.jpg") for k in range(N)]
    return assemble_timeline(video_id="Х-тест", duration_sec=TIMES[-1], frame_size=(464, 848), frames=frames,
                             tracks=_tracks(), objects=[[] for _ in range(N)],
                             zones={"desk": DESK, "door": DOOR, "safe": None, "laptop": None},
                             state_samples={}, config=config, person_pose=person_pose)


def test_flag_off_ignores_pose() -> None:
    cfg = SceneConfig(fps=FPS)
    plain = scene_to_dict(_timeline(cfg))
    assert scene_to_dict(_timeline(cfg, _person_pose())) == plain
    # флаг включён, но точек нет (прод без ViTPose) — сцена как без флага, кроме конфига в meta
    on = scene_to_dict(_timeline(dataclasses.replace(cfg, pose_departure=True)))
    on["meta"]["scene"]["config"] = plain["meta"]["scene"]["config"]
    assert on == plain


def test_raw_cache_keeps_person_pose(tmp_path: Path) -> None:
    frames = [FrameRef(k, TIMES[k], f"frames/{k:06d}.jpg") for k in range(N)]
    cfg = SceneConfig(fps=FPS)
    path = save_raw(tmp_path / "Х-тест.json", video_id="Х-тест", duration_sec=TIMES[-1], frame_size=(464, 848),
                    frames=frames, tracks=_tracks(), objects=[[] for _ in range(N)],
                    zones={"desk": DESK, "door": DOOR, "safe": None, "laptop": None}, zone_info={},
                    owl_keyframes=[], state_samples={}, config=cfg, person_pose=_person_pose())
    raw = load_raw(path)
    assert len(raw["person_pose"]) == len(_person_pose()) and len(raw["person_pose"][0][2]) == 17
    base = rebuild_scene(raw, cfg)
    on = rebuild_scene(raw, dataclasses.replace(cfg, pose_departure=True))
    assert base.meta["scene"]["left_workplace_t"] == [TIMES[8]]           # ложный уход за стеклом
    assert on.meta["scene"]["left_workplace_t"] == []


def test_add_pose_to_raw_and_set_override(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from add_pose_to_raw import main as add_main
    from rebuild_from_raw import parse_overrides

    raw_dir, pose_dir, out_dir = tmp_path / "raw", tmp_path / "pose", tmp_path / "out"
    raw_dir.mkdir(), pose_dir.mkdir()
    raw = {"raw_version": 1, "frames": [[k, TIMES[k], f"f{k}.jpg"] for k in range(3)], "config": {}}
    (raw_dir / "X-7.json").write_text(json.dumps(raw), encoding="utf-8")         # латинская X
    states = [{"t": TIMES[k] + 1e-4, "cashier_bbox": list(SEATED), "cashier_kp": SEATED_KP if k != 1 else None,
               "others": [{"track_id": 4, "bbox": list(GLASS)}], "others_kp": [{"track_id": 4, "kp": GLASS_KP}]}
              for k in range(3)]
    (pose_dir / "Х-7.json").write_text(json.dumps({"states": states}), encoding="utf-8")   # кириллическая Х
    assert add_main(["--raw", str(raw_dir), "--pose-cache", str(pose_dir), "--out", str(out_dir)]) == 0
    got = json.loads((out_dir / "X-7.json").read_text(encoding="utf-8"))["person_pose"]
    assert [e[0] for e in got] == [0, 0, 1, 2, 2]                                # кассир без кадра 1 + others
    assert parse_overrides(["pose_departure=true", "pose_leg_score=0.3"]) == {"pose_departure": True,
                                                                               "pose_leg_score": 0.3}
