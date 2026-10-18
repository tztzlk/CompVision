"""Выбор зоны двери по сигналу прохода (worker.perception.door.pick_door_by_signal, SceneConfig.door_pick_signal):
соперник с проходом заменяет неподвижную панель, слабый сигнал / нет человека рядом / вложенная группа — нет;
без door_select и с выключенным флагом выбор прежний; пересчёт сырого кэша scripts/redoor_raw.py."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from worker.perception import Detection, Frame
from worker.perception.door import distinct_groups, door_activity, pick_door_by_signal
from worker.perception.tracking import Track, TrackPoint
from worker.perception.zones import door_groups, infer_zones
from worker.scene.builder import SceneConfig
from worker.scene.pipeline import door_selector, door_signal, select_zones

ROOT = Path(__file__).resolve().parents[1]
PANEL = (0.5, 0.0, 0.9, 0.5)        # неподвижная стеклянная панель — OWLv2 уверен в ней больше
DOOR = (0.05, 0.0, 0.45, 0.5)       # дверь рядом
NESTED = (0.1, 0.0, 0.3, 0.45)      # узкая полоса внутри двери
NEAR_DOOR = (0.40, 0.2, 0.48, 0.7)  # человек у края двери в кадрах прохода
FAR = (0.0, 0.8, 0.1, 1.0)
N = 30                              # 10 с при 3 кадрах/с
PASS = range(12, 22)                # кадры, когда дверь открыта


def _owl(groups=((PANEL, "a glass door", 0.5), (DOOR, "a door", 0.4)), keyframes: int = 4) -> list[list[Detection]]:
    return [[Detection(label, score, box) for box, label, score in groups] for _ in range(keyframes)]


def _diffs(peak: float, at: range = PASS) -> list:
    return [peak if i in at else 0.02 for i in range(N)]


# ---------- правило на готовых door_diff ----------

def test_rival_with_passage_replaces_panel():
    persons = [[NEAR_DOOR] if i in range(10, 15) else [] for i in range(N)]
    k, info = pick_door_by_signal([PANEL, DOOR], lambda boxes: [_diffs(0.05), _diffs(0.9)][:len(boxes)], persons)
    assert k == 1 and info["replaced"] is True and info["pick"] == 1
    assert info["act"] == [pytest.approx(0.05), pytest.approx(0.9)]


@pytest.mark.parametrize("panel, door, persons, why", [
    (0.05, 0.12, [[NEAR_DOOR]] * N, "слабый сигнал: меньше min_act"),
    (0.6, 0.7, [[NEAR_DOOR]] * N, "меньше ratio × активности панели"),
    (0.05, 0.9, [[FAR]] * N, "в пике никого рядом — экран или свет"),
    (0.05, 0.9, [[NEAR_DOOR] if i < 3 else [] for i in range(N)], "человек рядом, но не в ± sync кадрах от пика"),
])
def test_rival_does_not_replace(panel, door, persons, why):
    k, info = pick_door_by_signal([PANEL, DOOR], lambda boxes: [_diffs(panel), _diffs(door)], persons)
    assert k == 0 and info["replaced"] is False, why


def test_single_spike_is_damped_by_median():
    spike = [0.02] * N
    spike[15] = 1.0                                     # один кадр-выброс (человек закрашен не целиком)
    assert door_activity(spike, 3)[0] < 0.1
    assert door_activity(spike, 1) == (1.0, 15)
    assert door_activity([None] * N, 3) == (0.0, None)


def test_nested_groups_are_not_rivals():
    assert distinct_groups([DOOR, NESTED, PANEL], topk=3, nest=0.5) == [0, 2]
    assert distinct_groups([DOOR, PANEL, (0.0, 0.0, 1.0, 1.0)], topk=3, nest=0.5) == [0, 1]   # «дверь + панель»
    assert distinct_groups([DOOR, PANEL, (0.92, 0.0, 1.0, 0.4)], topk=2, nest=0.5) == [0, 1]  # только topk
    seen = []

    def diffs_for(boxes):
        seen.append(list(boxes))
        return [_diffs(0.05) for _ in boxes]

    k, info = pick_door_by_signal([PANEL, (0.55, 0.0, 0.85, 0.45)], diffs_for, [[NEAR_DOOR]] * N)
    assert k == 0 and info["replaced"] is False and seen == []    # соперников нет — сигнал не считается


# ---------- infer_zones / select_zones ----------

def test_infer_zones_without_select_is_unchanged():
    owl = _owl()
    zones, info = infer_zones(owl, [])
    assert zones["door"] == PANEL and door_groups(owl)[0][0] == PANEL
    same, same_info = infer_zones(owl, [], door_select=lambda groups: 0)
    assert same == zones and same_info["door"] == info["door"]


def test_door_select_moves_door_before_safe():
    # кандидат сейфа внутри настоящей двери: пока дверь — панель, он сейф; дверь сменилась — отброшен (pick_safe)
    safe_in_door = (0.1, 0.25, 0.3, 0.45)
    owl = _owl(((PANEL, "a glass door", 0.5), (DOOR, "a door", 0.4), (safe_in_door, "a metal safe", 0.6)))
    zones, _ = infer_zones(owl, [])
    assert zones["safe"] == safe_in_door
    zones, info = infer_zones(owl, [], door_select=lambda groups: (1, {"signal": {"replaced": True}}))
    assert zones["door"] == DOOR and zones["safe"] is None
    assert info["door"]["signal"] == {"replaced": True} and info["door"]["score"] == pytest.approx(0.4)
    assert "door" in str(info["safe"]["reason"])


def test_enabled_by_default_flag_off_keeps_old_choice():
    assert SceneConfig().door_pick_signal is True
    tracks = [Track(1, [TrackPoint(0, 0.0, (0.4, 0.4, 0.7, 0.9), 0.9)])]
    zones, _ = select_zones(_owl(), [], tracks, SceneConfig(door_pick_signal=False))
    assert zones["door"] == PANEL


# ---------- сигнал по кадрам ----------

def _frames(tmp_path: Path, door_opens: bool = True) -> tuple[list[Frame], list[list[Detection]]]:
    import cv2

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    frames, persons = [], []
    x1, x2, y2 = int(DOOR[0] * 200), int(DOOR[2] * 200), int(DOOR[3] * 200)
    for i in range(N):
        img = base.copy()
        if door_opens and i in PASS:
            img[0:y2, x1:x2] = other[0:y2, x1:x2]          # в проёме открытой двери другой вид
        path = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(Frame(index=i, timestamp_sec=round(i / 3, 3), path=path))
        persons.append([Detection("person", 0.9, NEAR_DOOR)] if i in range(10, 15) else [])
    return frames, persons


def test_selector_picks_door_by_frames_and_caches_signal(tmp_path):
    frames, persons = _frames(tmp_path)
    cfg = dataclasses.replace(SceneConfig(), door_pick_signal=True)
    select, cache = door_selector(frames, persons, cfg)
    tracks = [Track(1, [TrackPoint(0, 0.0, (0.4, 0.6, 0.7, 0.95), 0.9)])]
    zones, info = select_zones(_owl(), [], tracks, cfg, select)
    sig = info["door"]["signal"]
    assert zones["door"] == DOOR and sig["replaced"] is True
    assert sig["act"][1] > 0.5 and sig["act"][0] < 0.1
    assert cache[tuple(DOOR)] == door_signal(frames, persons, DOOR, cfg)     # тот же door_diff, без пересчёта


def test_selector_keeps_panel_without_passage(tmp_path):
    frames, persons = _frames(tmp_path, door_opens=False)
    cfg = dataclasses.replace(SceneConfig(), door_pick_signal=True)
    select, _ = door_selector(frames, persons, cfg)
    zones, info = select_zones(_owl(), [], [], cfg, select)
    assert zones["door"] == PANEL and info["door"]["signal"]["replaced"] is False


# ---------- scripts/redoor_raw.py ----------

def _load_redoor():
    spec = importlib.util.spec_from_file_location("redoor_raw", ROOT / "scripts" / "redoor_raw.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_redoor_raw(tmp_path):
    fdir = tmp_path / "frames" / "Х-синт"
    fdir.mkdir(parents=True)
    frames, _ = _frames(fdir)
    raw = {
        "raw_version": 1, "video_id": "Х-синт", "duration_sec": 10.0, "frame_size": [200, 200],
        "frames": [[f.index, f.timestamp_sec, f"frames/{f.path.name}"] for f in frames],
        "tracks": [{"id": 1, "points": [[i, round(i / 3, 3), list(NEAR_DOOR), 0.9] for i in range(10, 15)]}],
        "objects": [[] for _ in frames],
        "owl_keyframes": [[[d.label, d.score, list(d.bbox_xyxy)] for d in dets] for dets in _owl()],
        "zones": {"desk": None, "safe": None, "door": list(PANEL), "laptop": None},
        "zone_info": {"door": {"source": "owlv2", "score": 0.5}},
        "state_samples": {}, "config": {}, "meta": {}, "door_diff": [0.0] * N, "person_reid": [["kept"]],
    }
    mod = _load_redoor()
    cfg = dataclasses.replace(SceneConfig(), door_pick_signal=True)
    out = mod.redoor(raw, fdir, cfg)
    assert out["zones"]["door"] == list(DOOR) and out["meta"]["redoor"]["door_changed"] is True
    assert out["meta"]["redoor"]["safe_changed"] is False and out["person_reid"] == [["kept"]]
    assert max(d for d in out["door_diff"] if d is not None) > 0.5
    assert out["zone_info"]["door"]["signal"]["replaced"] is True
    off = mod.redoor(raw, fdir, dataclasses.replace(cfg, door_pick_signal=False))
    assert off["zones"]["door"] == list(PANEL) and off["door_diff"] == raw["door_diff"]
    assert off["meta"]["redoor"]["door_changed"] is False


# ---------- --rezone на кэше redoor_raw: дверь та же, для которой посчитан door_diff ----------

def _rebuild_cli():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import rebuild_from_raw

    return rebuild_from_raw


def _redoored(tmp_path) -> dict:
    fdir = tmp_path / "frames" / "Х-синт"
    fdir.mkdir(parents=True)
    frames, _ = _frames(fdir)
    raw = {
        "raw_version": 1, "video_id": "Х-синт", "duration_sec": 10.0, "frame_size": [200, 200],
        "frames": [[f.index, f.timestamp_sec, f"frames/{f.path.name}"] for f in frames],
        "tracks": [{"id": 1, "points": [[i, round(i / 3, 3), list(NEAR_DOOR), 0.9] for i in range(10, 15)]}],
        "objects": [[] for _ in frames],
        "owl_keyframes": [[[d.label, d.score, list(d.bbox_xyxy)] for d in dets] for dets in _owl()],
        "zones": {"desk": None, "safe": None, "door": list(PANEL), "laptop": None},
        "zone_info": {"door": {"source": "owlv2", "score": 0.5}},
        "state_samples": {}, "config": {"fps": 3.0}, "meta": {}, "door_diff": [0.0] * N,
    }
    return _load_redoor().redoor(raw, fdir, dataclasses.replace(SceneConfig(), fps=3.0, door_pick_signal=True))


def _door(scene) -> tuple:
    z = scene.zones["door"]
    return tuple(getattr(z, "bbox", z))


def test_redoor_records_signal_in_config(tmp_path):
    out = _redoored(tmp_path)
    assert out["config"]["door_pick_signal"] is True and out["config"]["fps"] == 3.0
    assert "door_pick_signal" in _rebuild_cli().PERCEPTION_FIELDS


@pytest.mark.parametrize("code_config", [False, True])
def test_rezone_keeps_signal_door(tmp_path, code_config):
    """rebuild_scene(rezone=True): с конфигом кэша и с --code-config (PERCEPTION_FIELDS) дверь — та, по сигналу."""
    from worker.scene.raw_cache import rebuild_scene

    out = _redoored(tmp_path)
    config = None
    if code_config:
        keep = {k: out["config"][k] for k in _rebuild_cli().PERCEPTION_FIELDS if k in out["config"]}
        config = dataclasses.replace(SceneConfig(), **keep)
        assert config.door_pick_signal is True
    scene = rebuild_scene(out, config, rezone=True)
    assert _door(scene) == pytest.approx(DOOR) and "door_diff_stale" not in scene.meta


def test_rezone_marks_stale_door_diff(tmp_path):
    """Без door_pick_signal rezone берёт панель (лучшая по сумме), а door_diff — от двери: сцена помечена."""
    from worker.scene.raw_cache import rebuild_scene

    out = _redoored(tmp_path)
    out["config"]["door_pick_signal"] = False
    scene = rebuild_scene(out, None, rezone=True)
    assert _door(scene) == pytest.approx(PANEL)
    assert scene.meta["door_diff_stale"]["cached_door"] == pytest.approx(list(DOOR))
    assert "door_diff_stale" not in rebuild_scene(out, None, rezone=False).meta
