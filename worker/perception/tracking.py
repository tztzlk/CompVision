"""Трекер людей по IoU и расстоянию между центрами рамок (только stdlib).

Почему не ByteTrack: в кадре кассы 1–4 человека, анализ идёт на 2–3 к/с. Жадного сопоставления
с прогнозом по скорости достаточно, а свой трекер не тянет зависимостей и тестируется без GPU.
Разрывы треков (человек пропал на 1–2 кадра у двери) склеиваются позже, в worker.scene.builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from worker.perception.geometry import BBox, center, center_dist, clip, iou

__all__ = ["TrackPoint", "Track", "IoUTracker"]


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int        # позиция кадра в списке проанализированных кадров
    t: float                # секунды от начала видео
    bbox: BBox
    score: float


@dataclass
class Track:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def first_t(self) -> float:
        return self.points[0].t

    @property
    def last_t(self) -> float:
        return self.points[-1].t

    @property
    def duration(self) -> float:
        return self.last_t - self.first_t


class IoUTracker:
    """Жадное сопоставление рамок кадра с активными треками.

    Пара «трек–рамка» — кандидат, если IoU с прогнозом трека >= iou_threshold или расстояние
    между центрами <= max_center_dist. Кандидаты разбираются по убыванию (IoU - расстояние).
    Трек без совпадений живёт max_lost_sec, затем закрывается. Несопоставленная рамка открывает
    новый трек, если её score >= min_new_score.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.2,
        max_center_dist: float = 0.12,
        max_lost_sec: float = 1.5,
        min_new_score: float = 0.5,
        velocity_damping: float = 0.5,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_center_dist = max_center_dist
        self._max_lost_sec = max_lost_sec
        self._min_new_score = min_new_score
        self._damping = velocity_damping
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """Все треки (активные и закрытые) в порядке появления."""
        return [tr for tr in self._tracks if tr.points]

    def _predict(self, track: Track, t: float) -> BBox:
        """Положение трека в момент t: последняя рамка, сдвинутая по затухающей скорости."""
        last = track.points[-1]
        if len(track.points) < 2 or self._damping <= 0:
            return last.bbox
        prev = track.points[-2]
        dt = last.t - prev.t
        if dt <= 0:
            return last.bbox
        (lx, ly), (px, py) = center(last.bbox), center(prev.bbox)
        k = self._damping * (t - last.t) / dt
        dx, dy = (lx - px) * k, (ly - py) * k
        b = last.bbox
        return clip((b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy))

    def update(self, frame_index: int, t: float, boxes: Sequence[tuple[BBox, float]]) -> list[Optional[int]]:
        """Сопоставить рамки кадра (bbox, score) с треками. Возвращает track_id для каждой рамки
        (None — рамка отброшена: не совпала ни с одним треком и слишком слабая для нового)."""
        active = [tr for tr in self._tracks if tr.last_t < t and t - tr.last_t <= self._max_lost_sec]
        predicted = [self._predict(tr, t) for tr in active]
        candidates: list[tuple[float, int, int]] = []
        for ti, pred in enumerate(predicted):
            for di, (box, _score) in enumerate(boxes):
                overlap = iou(pred, box)
                dist = center_dist(pred, box)
                if overlap >= self._iou_threshold or dist <= self._max_center_dist:
                    candidates.append((overlap - dist, ti, di))
        candidates.sort(reverse=True)

        assigned: list[Optional[int]] = [None] * len(boxes)
        used: set[int] = set()
        for _, ti, di in candidates:
            if ti in used or assigned[di] is not None:
                continue
            used.add(ti)
            track = active[ti]
            box, score = boxes[di]
            track.points.append(TrackPoint(frame_index, t, box, score))
            assigned[di] = track.track_id

        for di, (box, score) in enumerate(boxes):
            if assigned[di] is None and score >= self._min_new_score:
                track = Track(self._next_id)
                self._next_id += 1
                track.points.append(TrackPoint(frame_index, t, box, score))
                self._tracks.append(track)
                assigned[di] = track.track_id
        return assigned
