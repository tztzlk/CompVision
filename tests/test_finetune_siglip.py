"""scripts/finetune_siglip.py без видеокарты и весов: метки, маскированный лосс, аугментации, предобработка."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import finetune_siglip as ft  # noqa: E402


def test_desk_labels_columns_and_override(tmp_path):
    path = tmp_path / "desk.csv"
    path.write_text("# комментарий\nvideo_id,t_from,t_to,cash,bag,coat,cup\n"
                    "Х-4,0,999,0,0,0,0\nХ-4,12,47.5,?,0,0,0\nХ-4,48,999,1,0,0,0\n", encoding="utf-8")
    segs = ft.desk_labels(path)["X-4"]
    assert ft.labels_at(segs, 5.0) == {"desk_cash": 0.0, "desk_bag": 0.0, "desk_coat": 0.0, "desk_cup": 0.0}
    assert np.isnan(ft.labels_at(segs, 20.0)["desk_cash"]) and ft.labels_at(segs, 20.0)["desk_bag"] == 0.0
    assert ft.labels_at(segs, 50.0)["desk_cash"] == 1.0


def test_desk_labels_item_list(tmp_path):
    path = tmp_path / "desk.csv"
    path.write_text("video_id,t_from,t_to,items\nX-1,0,10,cash;bag\nX-1,5,6,cup?\nX-1,8,9,?\n", encoding="utf-8")
    segs = ft.desk_labels(path)["X-1"]
    a = ft.labels_at(segs, 1.0)
    assert a["desk_cash"] == 1.0 and a["desk_bag"] == 1.0 and a["desk_cup"] == 0.0
    b = ft.labels_at(segs, 5.5)
    assert np.isnan(b["desk_cup"]) and b["desk_cash"] == 0.0
    assert all(np.isnan(v) for v in ft.labels_at(segs, 8.5).values())


def test_safe_labels_occluded_masked(tmp_path):
    path = tmp_path / "safe.csv"
    path.write_text("video_id,t_from,t_to,state\nХ-5,0,999,closed\nХ-5,47,47.5,occluded\nХ-5,48,50.5,open\n",
                    encoding="utf-8")
    segs = ft.safe_labels(path)["X-5"]
    assert ft.labels_at(segs, 10)["safe_open"] == 0.0
    assert np.isnan(ft.labels_at(segs, 47.2)["safe_open"])
    assert ft.labels_at(segs, 49)["safe_open"] == 1.0


def test_masked_bce_ignores_nan_and_weights():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 100.0], [-2.0, -100.0]])
    y = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])
    one = torch.ones(2)
    loss = ft.masked_bce(logits, y, one, one)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([2.0, -2.0]), torch.tensor([1.0, 0.0]))
    assert torch.isfinite(loss) and abs(float(loss) - float(ref)) < 1e-6


def test_class_weights_like_fit_logreg():
    y = np.array([[1, np.nan], [0, 1], [0, 0], [0, np.nan]], dtype=np.float32)
    pw, nw = ft.class_weights(y)
    assert pw[0] == pytest.approx(4 / 2) and nw[0] == pytest.approx(4 / 6)
    assert pw[1] == pytest.approx(1.0) and nw[1] == pytest.approx(1.0)


def test_augment_shape_and_identity_box():
    ctx = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    rel = np.float32([0.1, 0.1, 0.9, 0.9])
    for seed in range(5):
        out = ft.augment(ctx, rel, random.Random(seed))
        assert out.shape == (224, 224, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]


def test_to_pixels_matches_siglip_normalization():
    pytest.importorskip("torch")
    rgb = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    rgb[0, :, :, 0] = 255
    x = ft.to_pixels(rgb)
    assert x.shape == (1, 3, 224, 224)
    assert float(x[0, 0].min()) == 1.0 and float(x[0, 1].max()) == -1.0
