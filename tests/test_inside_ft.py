"""Башня «внутри кассы» (SceneConfig.inside_ft): точки за стеклом не «у места» и не люди в кассе — без моделей."""

from __future__ import annotations

from worker.perception.tracking import Track, TrackPoint
from worker.scene.builder import FrameRef, SceneConfig, assemble_timeline

DESK = (0.10, 0.45, 0.70, 0.70)
SEATED = (0.30, 0.30, 0.55, 0.75)
CORRIDOR = (0.60, 0.20, 0.66, 0.47)            # в коридоре за стеклом: низ ниже верха стола — геометрия «внутри»
FAR = (0.75, 0.20, 0.95, 0.75)                 # кассир ушла от места
GLASS = (0.34, 0.33, 0.51, 0.62)               # за стеклом прямо над рабочим местом, score высокий (Х-11, Х-27)
ON = SceneConfig(fps=3.0, inside_ft="ft/person_inside")


def _timeline(boxes_by_track: dict[int, list], p_by_box: dict, config: SceneConfig, with_ft: bool = True):
    n = max(len(b) for b in boxes_by_track.values())
    times = [round(k / 3, 3) for k in range(n)]
    tracks = [Track(tid, [TrackPoint(k, times[k], b, 0.9) for k, b in enumerate(boxes) if b is not None])
              for tid, boxes in boxes_by_track.items()]
    person_ft = [(p.frame_index, p.bbox, p_by_box.get(p.bbox, 0.99)) for t in tracks for p in t.points] if with_ft else None
    return assemble_timeline(
        video_id="Х-тест", duration_sec=times[-1], frame_size=(464, 848),
        frames=[FrameRef(k, times[k], f"frames/{k:06d}.jpg") for k in range(n)], tracks=tracks,
        objects=[[] for _ in range(n)], zones={"desk": DESK, "door": None, "safe": None, "laptop": None},
        state_samples={}, config=config, person_ft=person_ft)


def _states(scene):
    return [(s.persons_in_room, s.cashier_bbox, s.value("cashier_away"), s.value("cashier_left_workplace"))
            for s in scene.states]


def test_flag_off_ignores_person_ft():
    boxes = {1: [SEATED] * 12, 3: [CORRIDOR] * 12}
    off = SceneConfig(fps=3.0)
    assert _states(_timeline(boxes, {CORRIDOR: 0.01}, off)) == _states(_timeline(boxes, {}, off, with_ft=False))


def test_corridor_point_not_counted_in_room():
    boxes = {1: [SEATED] * 12, 3: [CORRIDOR] * 12}
    assert _timeline(boxes, {}, ON, with_ft=False).states[6].persons_in_room == 2      # без башни — по геометрии
    scene = _timeline(boxes, {CORRIDOR: 0.03}, ON)
    assert scene.states[6].persons_in_room == 1
    assert not scene.states[6].others


def test_unsure_tower_keeps_geometry():
    boxes = {1: [SEATED] * 12, 3: [CORRIDOR] * 12}
    assert _timeline(boxes, {CORRIDOR: 0.5}, ON).states[6].persons_in_room == 2


def test_behind_glass_over_workplace_is_not_return():
    # сидит 6 с, уходит вдаль на 2 с, дальше 4 с за стеклом над местом со score 0.9 (Х-11, Х-27)
    boxes = {1: [SEATED] * 18 + [FAR] * 6 + [GLASS] * 12}
    base = _timeline(boxes, {}, ON, with_ft=False)
    assert not any(s.value("cashier_left_workplace") for s in base.states)   # без башни — «вернулась» через 2 с
    scene = _timeline(boxes, {GLASS: 0.05}, ON)
    assert any(s.value("cashier_left_workplace") for s in scene.states)
    assert scene.states[-1].value("cashier_away")                            # с башней — так и не вернулась
    assert scene.states[-1].cashier_bbox is None and scene.states[-1].persons_in_room == 0


DOOR = (0.80, 0.10, 0.98, 0.80)
BEHIND_DOOR = (0.82, 0.20, 0.96, 0.75)         # за стеклянной дверью в коридоре (геометрия «внутри»), p = 0.05
IN_DOORWAY = (0.78, 0.25, 0.93, 0.78)          # у двери изнутри, p = 0.99


def _door_scene(mode: str):
    # 2 с за дверью до входа, 1 с у двери, 4 с на месте, 1 с у двери, 2 с за дверью после выхода
    boxes = [BEHIND_DOOR] * 6 + [IN_DOORWAY] * 3 + [SEATED] * 12 + [IN_DOORWAY] * 3 + [BEHIND_DOOR] * 6
    n = len(boxes)
    times = [round(k / 3, 3) for k in range(n)]
    tracks = [Track(1, [TrackPoint(k, times[k], b, 0.9) for k, b in enumerate(boxes)])]
    person_ft = [(p.frame_index, p.bbox, 0.05 if p.bbox == BEHIND_DOOR else 0.99) for p in tracks[0].points]
    return assemble_timeline(
        video_id="Х-тест", duration_sec=times[-1] + 3, frame_size=(464, 848),
        frames=[FrameRef(k, times[k], f"frames/{k:06d}.jpg") for k in range(n)], tracks=tracks,
        objects=[[] for _ in range(n)], zones={"desk": DESK, "door": DOOR, "safe": None, "laptop": None},
        state_samples={}, config=SceneConfig(fps=3.0, inside_ft="ft/person_inside", inside_ft_door=mode),
        person_ft=person_ft)


def _exits(scene) -> list[int]:
    return [k for k, s in enumerate(scene.states) if s.value("cashier_left_room")]


def test_door_zone_untouched_by_default():
    scene = _door_scene("")
    assert _exits(scene) == [29]                          # выход — последняя точка за дверью (поздно)
    assert scene.states[1].persons_in_room == 1


def test_door_exit_mode_trims_tail_keeps_entry():
    scene = _door_scene("exit")
    assert _exits(scene) == [23]                          # последняя точка внутри у двери
    assert scene.states[1].persons_in_room == 1          # до входа за дверью — как по геометрии (вход не трогаем)
    assert _door_scene("all").states[1].persons_in_room == 0
