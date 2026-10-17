"""Генератор фикстур tests/fixtures/scene_*.json по контракту SceneTimeline v1 (раздел 5 задания B).

Сценарии собраны вручную по описанию роликов (раздел 3), значения условные.
Запуск из корня: python tests/fixtures/make_scene_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

FPS = 3.0
ZONES = {
    "desk": [0.05, 0.38, 0.62, 0.62],
    "workplace": [0.20, 0.45, 0.75, 0.80],
    "safe": [0.62, 0.28, 0.82, 0.55],
    "door": [0.00, 0.05, 0.30, 0.50],
    "laptop": [0.45, 0.30, 0.60, 0.40],
}
CASHIER_BBOX = [0.30, 0.40, 0.60, 0.85]
FIELDS = ("safe_open", "door_open", "cash_on_desk", "bag_on_desk", "collection_bag",
          "outerwear_on_desk", "cup_or_food", "phone_in_hand", "laptop_unlocked")


def timeline(video_id: str, duration: float, state_at) -> dict:
    states = []
    n = int(duration * FPS)
    for i in range(n):
        t = round(i / FPS, 3)
        s = {
            "t": t, "frame": f"frames/{i:06d}.jpg", "persons_in_room": 1,
            "cashier_present": 0.95, "cashier_bbox": CASHIER_BBOX, "others": [],
            "person_entered": False, "person_exited": False,
            **{f: 0.02 for f in FIELDS},
        }
        s["laptop_unlocked"] = None
        s["collection_bag"] = None
        s.update(state_at(t))
        if s["cashier_present"] is not None and s["cashier_present"] < 0.5 and "cashier_bbox" not in state_at(t):
            s["cashier_bbox"] = None
        states.append(s)
    return {
        "version": 1, "video_id": video_id, "duration_sec": duration, "fps_sampled": FPS,
        "frame_size": [464, 848], "zones": ZONES, "states": states,
        "meta": {"device": "fixture", "elapsed_sec": 0.0, "models": {}},
    }


def x20(t: float) -> dict:
    # кассир достаёт деньги из сейфа (3–5 с), дверца остаётся открытой, с 5 с кассир за столом
    at_safe = 3.0 <= t < 5.0
    return {"safe_open": 0.9 if t >= 3.0 else 0.03, "cash_on_desk": 0.8 if t >= 6.0 else 0.05,
            "cashier_bbox": [0.60, 0.30, 0.80, 0.60] if at_safe else CASHIER_BBOX,
            "cashier_present": 0.3 if at_safe else 0.95}


def x40(t: float) -> dict:
    # с 17 с на столе пальто и сумка
    return {"outerwear_on_desk": 0.85 if t >= 17.0 else 0.02, "bag_on_desk": 0.75 if t >= 17.3 else 0.02}


def x15(t: float) -> dict:
    # сумка на столе всё время; с 15 с кассир уходит к двери
    away = t >= 15.0
    return {"bag_on_desk": 0.8, "cashier_present": 0.1 if away else 0.95,
            "person_exited": 16.0 <= t < 16.4}


def x18(t: float) -> dict:
    # клиент в комнате с начала; на 20 с через дверь входит третий человек (трек 7)
    others = [{"track_id": 3, "bbox": [0.35, 0.20, 0.55, 0.45], "first_seen_t": 0.0}]
    if t >= 19.9:
        others.append({"track_id": 7, "bbox": [0.05, 0.10, 0.25, 0.55], "first_seen_t": 20.0})
    entered = 19.9 <= t < 20.1
    return {"persons_in_room": 1 + len(others), "others": others, "person_entered": entered,
            "visitor_entered_ids": [7] if entered else [], "cash_on_desk": 0.05,
            "door_open": 0.8 if 19.0 <= t < 21.0 else 0.05}


CLIENT_SEAT = [0.15, 0.15, 0.35, 0.45]      # напротив кассира, по другую сторону стола


def x3(t: float) -> dict:
    # трудный негатив: на 4 с клиент входит и садится напротив кассира; на столе пусто
    others = [{"track_id": 4, "bbox": CLIENT_SEAT, "first_seen_t": 4.0}] if t >= 3.9 else []
    entered = 3.9 <= t < 4.1
    return {"persons_in_room": 1 + len(others), "others": others, "person_entered": entered,
            "visitor_entered_ids": [4] if entered else [], "cash_on_desk": 0.03}


def x4(t: float) -> dict:
    # на 52 с предыдущий клиент уходит, на 57 с входит следующий и садится напротив; на столе наличные
    others = []
    if t < 52.0:
        others.append({"track_id": 2, "bbox": CLIENT_SEAT, "first_seen_t": 0.0})
    if t >= 56.9:
        others.append({"track_id": 9, "bbox": CLIENT_SEAT, "first_seen_t": 57.0})
    entered = 56.9 <= t < 57.1
    return {"persons_in_room": 1 + len(others), "others": others, "person_entered": entered,
            "person_exited": 51.9 <= t < 52.1, "visitor_entered_ids": [9] if entered else [],
            "cash_on_desk": 0.75 if 40.0 <= t < 66.0 else 0.05}


def x30(t: float) -> dict:
    # кассир один; на 12,5 с входит человек и встаёт рядом с кассиром, на его стороне стола
    others = [{"track_id": 6, "bbox": [0.33, 0.40, 0.57, 0.80], "first_seen_t": 12.667}] if t >= 12.6 else []
    entered = 12.6 <= t < 12.7
    return {"persons_in_room": 1 + len(others), "others": others, "person_entered": entered,
            "visitor_entered_ids": [6] if entered else [], "cash_on_desk": 0.05}


def x23(t: float) -> dict:
    # трудный негатив: кассир идёт к сейфу (8–12 с), сейф закрыт, на столе пусто
    at_safe = 8.0 <= t < 12.0
    return {"cashier_present": 0.2 if at_safe else 0.95, "safe_open": 0.03,
            "cashier_bbox": [0.60, 0.30, 0.80, 0.60] if at_safe else CASHIER_BBOX}


SCENES = {
    "scene_X20.json": timeline("Х-20", 15.0, x20),
    "scene_X40.json": timeline("Х-40", 25.0, x40),
    "scene_X15.json": timeline("Х-15", 22.0, x15),
    "scene_X18.json": timeline("Х-18", 26.0, x18),
    "scene_X23.json": timeline("Х-23", 20.0, x23),
    "scene_X3.json": timeline("Х-3", 20.0, x3),
    "scene_X4.json": timeline("Х-4", 72.0, x4),
    "scene_X30.json": timeline("Х-30", 20.0, x30),
}

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name, data in SCENES.items():
        (here / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, len(data["states"]))
