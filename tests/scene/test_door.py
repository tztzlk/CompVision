"""Сигнал двери для R15 (worker.perception.door) на синтетических кадрах: закрыта / открыта / снова закрыта,
сдвиг яркости, люди в проёме, нет эталона."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from worker.perception.door import DOOR_GRID, door_band, door_diffs, door_patch

WHOLE = (0.0, 0.0, 1.0, 1.0)            # зона двери — весь кадр
RNG = np.random.default_rng(7)
CLOSED = RNG.uniform(40, 220, size=(240, 120))
OPENED = CLOSED.copy()
OPENED[:, 40:] = RNG.uniform(0, 60, size=(240, 80))     # створка ушла, в проёме тёмный коридор


def _img(gray: np.ndarray, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    g = np.clip(gray * gain + offset, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _diffs(images, boxes=None, **kw):
    boxes = boxes or [[] for _ in images]
    times = [round(i / 3.0, 3) for i in range(len(images))]
    return door_diffs([door_patch(im, WHOLE, b) for im, b in zip(images, boxes)], times, **kw)


def test_band_takes_upper_part_of_door():
    assert door_band((0.1, 0.2, 0.3, 0.6), 0.0, 0.5) == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_closed_open_closed():
    images = [_img(OPENED if 12 <= i < 24 else CLOSED) for i in range(36)]      # открыта с 4 до 8 с
    d = _diffs(images)
    assert max(d[:12]) < 0.05
    assert min(d[12:24]) > 0.5
    assert max(d[24:]) < 0.05


def test_brightness_and_contrast_shift_is_not_a_change():
    images = [_img(CLOSED)] * 6 + [_img(CLOSED, gain=0.6, offset=30)] * 6
    assert max(_diffs(images)) < 0.05


def test_person_cells_are_left_out():
    noisy = CLOSED.copy()
    noisy[:, :24] = RNG.uniform(0, 255, size=(240, 24))     # человек в левой пятой части проёма
    images = [_img(CLOSED)] * 6 + [_img(noisy)] * 6
    person = (0.0, 0.0, 0.2, 1.0)
    d = _diffs(images, [[]] * 6 + [[person]] * 6)
    assert max(d) < 0.05
    unmasked = _diffs(images)                                # без маски тот же кадр — заметная разница
    assert min(unmasked[6:]) > 0.1


def test_door_mostly_covered_by_person_has_no_value():
    big = (0.0, 0.0, 0.8, 1.0)
    d = _diffs([_img(CLOSED)] * 6, [[]] * 3 + [[big]] * 3)
    assert d[3:] == [None, None, None]
    assert d[0] == pytest.approx(0.0, abs=0.01)


def test_no_baseline_when_door_covered_at_start():
    big = (0.0, 0.0, 0.9, 1.0)
    d = _diffs([_img(CLOSED)] * 9, [[big]] * 4 + [[]] * 5, baseline_sec=1.0)
    assert d == [None] * 9


def test_mask_follows_person_box():
    gray, mask = door_patch(_img(CLOSED), WHOLE, [(0.5, 0.0, 1.0, 0.5)], margin=0.0)
    gw, gh = DOOR_GRID
    assert gray.shape == mask.shape == (gh, gw)
    assert mask[: gh // 2 - 1, gw // 2 + 1:].all()
    assert not mask[gh // 2 + 1:, :].any()
    assert not mask[:, : gw // 2 - 1].any()


def test_tiny_zone_and_length_mismatch():
    assert door_patch(_img(CLOSED), (0.0, 0.0, 0.01, 0.01), []) is None
    with pytest.raises(ValueError):
        door_diffs([None], [0.0, 0.3])
    assert door_diffs([], []) == []


def test_numpy_fallback_without_opencv(monkeypatch):
    with_cv = _diffs([_img(CLOSED)] * 4 + [_img(OPENED)] * 4)
    monkeypatch.setitem(sys.modules, "cv2", None)           # import cv2 -> ImportError
    without_cv = _diffs([_img(CLOSED)] * 4 + [_img(OPENED)] * 4)
    assert max(without_cv[:4]) < 0.05 and min(without_cv[4:]) > 0.5
    assert without_cv[5] == pytest.approx(with_cv[5], abs=0.1)
