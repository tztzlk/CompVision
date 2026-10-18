"""Флаг SceneConfig.safe_ft (дообученная башня SigLIP): по умолчанию выключен, без весов — сцена как без
флага, R10 берёт ft_* поверх head_*, scripts/apply_ft.py сопоставляет вероятности по мс."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from worker.events import detectors as det
from worker.perception.ft_tower import load_ft_tower
from worker.scene.builder import SceneConfig
from worker.scene.contract import load_scene

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import apply_ft  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def test_flag_off_by_default():
    assert SceneConfig().safe_ft == ""
    assert SceneConfig().safe_ft_fields == ("desk_cash",)
    assert "safe_ft" in SceneConfig().as_dict()


def test_missing_weights_error_not_exception(tmp_path):
    assert load_ft_tower(tmp_path, "", "cpu") == (None, None)
    tower, err = load_ft_tower(tmp_path, "ft/siglip_states", "cpu")
    assert tower is None and "siglip_states" in err


# --- R10: ft_* поверх head_* --------------------------------------------------------------------

def test_r10_with_ft_uses_ft_params(rules):
    """Башня с порогом ft_p_on 0,6 видит открытие с p 0,7, голова с head_p_on 0,8 — нет."""
    r10 = next(r for r in rules if r.id == "R10_storage_left_open")
    params = {**r10.params, "head_p_on": 0.8, "head_p_off": 0.5, "ft_p_on": 0.6, "ft_p_off": 0.4}
    scene = load_scene(FIXTURES / "scene_X23.json")
    late = [dataclasses.replace(s, safe_open=0.7 if s.t >= 2.0 else 0.02) for s in scene.states]
    head = dataclasses.replace(scene, states=tuple(late), meta={**scene.meta, "safe_head": {"name": "safe_open"}})
    ft = dataclasses.replace(head, meta={**head.meta, "safe_ft": {"name": "siglip_states"}})
    assert det.storage_left_open(head, params) == []
    assert [round(e.timecode_sec) for e in det.storage_left_open(ft, params)] == [12]
    # ошибка загрузки башни (meta.safe_ft без name) — параметры головы, как без флага
    failed = dataclasses.replace(head, meta={**head.meta, "safe_ft": {"error": "нет весов"}})
    assert det.storage_left_open(failed, params) == []


def test_r14_with_ft_cash_uses_ft_params_and_empty_window(rules):
    """R14 при meta.safe_ft.fields с desk_cash берёт ft_*; «пустое окно» — вход засчитывается, только если
    перед ним в кассе был один кассир."""
    r14 = next(r for r in rules if r.id.startswith("R14"))
    scene = load_scene(FIXTURES / "scene_X4.json")
    entries = det._merge_entries(scene, det._entries(scene), 2.0)
    assert entries
    t_in = scene.states[entries[-1][0]].t
    states = tuple(dataclasses.replace(s, extra={**s.extra, "desk_cash": 0.0,
                                                 "desk_cash_ft": 0.7 if abs(s.t - t_in) <= 2 else 0.0})
                   for s in scene.states)
    params = {**r14.params, "ft_cash_p": 0.6, "ft_empty_before_sec": 0.0}
    assert params["ft_field"] == "desk_cash_ft"
    owl = dataclasses.replace(scene, states=states)
    ft = dataclasses.replace(owl, meta={**scene.meta, "safe_ft": {"name": "siglip_states",
                                                                 "fields": ["safe_open", "desk_cash"]}})
    safe_only = dataclasses.replace(owl, meta={**scene.meta, "safe_ft": {"name": "siglip_states",
                                                                        "fields": ["safe_open"]}})
    assert det.entry_with_cash_on_desk(owl, params) == []
    assert det.entry_with_cash_on_desk(safe_only, params) == []
    hits = det.entry_with_cash_on_desk(ft, params)
    assert round(t_in, 1) in [round(e.timecode_sec, 1) for e in hits]
    crowded = dataclasses.replace(ft, states=tuple(dataclasses.replace(s, persons_in_room=2) for s in ft.states))
    assert det.entry_with_cash_on_desk(crowded, {**params, "ft_empty_before_sec": 6.0}) == []


def test_rules_have_ft_params(rules):
    r10 = next(r for r in rules if r.id == "R10_storage_left_open")
    assert {"ft_p_on", "ft_p_off"} <= set(r10.params)
    assert r10.params["ft_p_off"] < r10.params["ft_p_on"]


# --- scripts/apply_ft.py ------------------------------------------------------------------------

def _raw(vid: str, times: list[float]) -> dict:
    return {"video_id": vid, "zones": {"safe": [0.6, 0.2, 0.8, 0.5]}, "meta": {}, "config": {},
            "state_samples": {"safe_open": [[t, 0.3] for t in times], "cash_on_desk": [[t, 0.1] for t in times]}}


def test_apply_ft_matches_by_ms_not_rounded_seconds():
    # t из .npz — float32: 2.3349998 и 2.335 — одна выборка; 2.4 — другая, хотя round(t, 1) у 2.35 и 2.4 разный
    times = [0.0, 1.002, 2.335, 2.4]
    t32 = np.array([0.0, 1.002, 2.335], dtype=np.float32)
    probs = {int(round(float(t) * 1000)): p for t, p in zip(t32, (0.9, 0.8, 0.7))}
    raw = _raw("Х-7", times)
    counts = apply_ft.substitute(raw, probs)
    assert counts == {"replaced": 3, "filled": 0, "missing": 1}
    assert raw["state_samples"]["safe_open"] == [[0.0, 0.9], [1.002, 0.8], [2.335, 0.7], [2.4, None]]
    assert raw["state_samples"]["safe_open_zero_shot"][0] == [0.0, 0.3]
    assert raw["meta"]["safe_ft"]["name"] == "siglip_states"
    assert raw["config"]["safe_ft"] == apply_ft.FT_REL


def test_apply_ft_keeps_head_value_and_fills():
    raw = _raw("Х-7", [0.0, 1.0, 2.0])
    raw["state_samples"]["safe_open_zero_shot"] = [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]]   # кэш с головой
    counts = apply_ft.substitute(raw, {0: 0.9}, keep_missing=True)
    assert counts == {"replaced": 1, "filled": 2, "missing": 0}
    assert raw["state_samples"]["safe_open"] == [[0.0, 0.9], [1.0, 0.3], [2.0, 0.3]]
    assert raw["state_samples"]["safe_open_head"][1] == [1.0, 0.3]
    assert raw["state_samples"]["safe_open_zero_shot"][1] == [1.0, 0.5]


def test_apply_ft_video_without_oof_untouched():
    raw = _raw("Х-7", [0.0, 1.0])
    assert apply_ft.substitute(raw, {})["missing"] == 2
    assert "safe_ft" not in raw["meta"] and raw["state_samples"]["safe_open"][0] == [0.0, 0.3]


def test_apply_ft_desk_nearest_key_within_half_second():
    raw = _raw("Х-7", [0.0, 1.002, 2.004, 3.006])
    raw["state_samples"]["cash_on_desk"] = [[0.0, 0.1], [1.002, 0.1], [2.004, 0.1], [3.006, 0.1]]
    probs = {0: 0.2, 1400: 0.9, 3700: 0.5}          # 1,4 с — ближайший к 1,002 (398 мс); к 3,006 — 694 мс
    counts = apply_ft.substitute_desk(raw, probs)
    assert counts == {"replaced": 2, "missing": 2}
    assert raw["state_samples"]["desk_cash"] == [[0.0, 0.2], [1.002, 0.9], [2.004, None], [3.006, None]]
    assert raw["meta"]["safe_ft"]["fields"] == ["desk_cash"]
    apply_ft.substitute(raw, {0: 0.1})
    assert raw["meta"]["safe_ft"]["fields"] == ["desk_cash", "safe_open"]


def test_apply_ft_cli_with_oof_npz(tmp_path):
    raw_dir, out = tmp_path / "raw", tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "Х-7.json").write_text(json.dumps(_raw("Х-7", [0.0, 1.002, 2.335]), ensure_ascii=False), encoding="utf-8")
    (raw_dir / "Х-8.json").write_text(json.dumps(_raw("Х-8", [0.0, 1.0]), ensure_ascii=False), encoding="utf-8")
    t = np.array([0.0, 1.002, 2.335], dtype=np.float32)
    np.savez(tmp_path / "safe_open_orig.npz", video=np.array(["X-7"] * 3), t=t,
             t_ms=np.array([int(round(float(x) * 1000)) for x in t], dtype=np.int64),
             y=np.zeros(3, np.int8), p=np.array([0.1, 0.2, 0.95], np.float32), fold=np.zeros(3, np.int8))
    assert apply_ft.main(["--raw", str(raw_dir), "--oof", str(tmp_path / "safe_open_orig.npz"), "--out", str(out)]) == 0
    a = json.loads((out / "Х-7.json").read_text(encoding="utf-8"))
    assert [p for _, p in a["state_samples"]["safe_open"]] == [0.1, 0.2, 0.95]
    assert a["meta"]["safe_ft"]["oof"] is True
    b = json.loads((out / "Х-8.json").read_text(encoding="utf-8"))
    assert "safe_ft" not in b["meta"]


# --- реальная башня на CPU: те же ключи state.pt и та же предобработка, что у конвейера ---------------

def _siglip_dir() -> Path:
    for cand in (ROOT / "models", ROOT / "models" / "models"):
        if (cand / "hf" / "siglip-base-patch16-224" / "config.json").is_file():
            return cand
    pytest.skip("нет локальных весов SigLIP")


def test_tower_matches_pipeline_embeddings_on_cpu(tmp_path):
    """state.pt с весами базы (последние 2 блока) и головой w = e_k: вероятность = sigmoid(эмбеддинг[k]),
    эмбеддинг — как ZeroShotClassifier.image_embeddings (голова раунда 3) на тех же BGR-вырезках."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    models_dir = _siglip_dir()
    from transformers import SiglipVisionModel

    from worker.perception.models import ZeroShotClassifier, model_path

    base = model_path(models_dir, "zero_shot")
    vm = SiglipVisionModel.from_pretrained(base, local_files_only=True).vision_model
    state = {f"vm.{k}": v.half() for k, v in vm.state_dict().items()
             if k.startswith(("encoder.layers.10.", "encoder.layers.11.", "post_layernorm.", "head."))}
    w = torch.zeros(5, vm.config.hidden_size)
    w[0, 3] = 40.0
    state.update({"cls.weight": w.half(), "cls.bias": torch.zeros(5).half()})
    out = tmp_path / "ft" / "siglip_states"
    out.mkdir(parents=True)
    torch.save(state, out / "state.pt")
    (out / "meta.json").write_text(json.dumps({
        "trainable_blocks": 2, "all_tasks": ["safe_open", "desk_cash", "desk_bag", "desk_coat", "desk_cup"],
        "tasks": [{"name": "safe_open", "zone": "safe", "index": 0}]}), encoding="utf-8")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (60 + 7 * i, 90, 3), dtype=np.uint8) for i in range(3)]
    tower, err = load_ft_tower(models_dir, str(out), "cpu")
    assert err is None and tower.tasks == {"safe_open": 0} and tower.tasks_for_zone("safe") == ["safe_open"]
    p = tower(crops)["safe_open"]
    emb = ZeroShotClassifier(base, "cpu").image_embeddings(crops)
    ref = 1.0 / (1.0 + np.exp(-40.0 * emb[:, 3]))
    assert np.allclose(p, ref, atol=2e-3)
    assert load_ft_tower(models_dir, str(out), "cpu")[0] is tower          # загрузка один раз
    with pytest.raises(KeyError):
        tower(crops, ["desk_cash"])                                        # задача не обучена


def test_tower_rejects_foreign_keys(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    models_dir = _siglip_dir()
    out = tmp_path / "bad"
    out.mkdir()
    torch.save({"vm.encoder.layers.11.nope": torch.zeros(1), "cls.weight": torch.zeros(1, 768),
                "cls.bias": torch.zeros(1)}, out / "state.pt")
    (out / "meta.json").write_text(json.dumps({"tasks": [{"name": "safe_open", "index": 0}]}), encoding="utf-8")
    tower, err = load_ft_tower(models_dir, str(out), "cpu")
    assert tower is None and "nope" in err
