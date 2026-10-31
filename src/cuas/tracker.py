"""Multi-object tracker.

``IoUTracker`` is original code in this repository (Apache-2.0). Association
follows the two-stage idea from ByteTrack (Zhang et al., 2021): high-score
boxes are matched first, then low-score boxes are offered to tracks that are
still free. This file does not copy FoundationVision/ByteTrack (MIT) or
abewley/sort (GPL-3.0).

A ``supervision`` ByteTrack (MIT) or Roboflow ``trackers`` ByteTrack
(Apache-2.0) can replace this class later. Both speak the same idea: a list
of boxes in, a list of stable ids out. The default stays inside this repo so
the demo installs nothing copyleft and nothing beyond ``requirements.txt``.
"""

from __future__ import annotations

from cuas.boxes import Detection, Track


def _match(
    tracks: list[Track],
    detections: list[Detection],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            overlap = track.bbox.iou(det.bbox)
            if overlap >= iou_threshold:
                pairs.append((overlap, ti, di))
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _overlap, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    return matches


class IoUTracker:
    """Greedy two-stage IoU tracker with stable integer ids and center trails."""

    def __init__(
        self,
        high_thresh: float = 0.35,
        low_thresh: float = 0.1,
        iou_thresh: float = 0.2,
        max_age: int = 15,
        min_hits: int = 1,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        high_matches = _match(self._tracks, high, self.iou_thresh)
        matched_ids = {self._tracks[ti].track_id for ti, _di in high_matches}
        matched_high = {di for _ti, di in high_matches}
        self._consume(high_matches, high)

        still_free = [t for t in self._tracks if t.track_id not in matched_ids]
        low_local = _match(still_free, low, self.iou_thresh)
        id_to_index = {t.track_id: i for i, t in enumerate(self._tracks)}
        low_matches = [(id_to_index[still_free[ti].track_id], di) for ti, di in low_local]
        for ti, _di in low_local:
            matched_ids.add(still_free[ti].track_id)
        self._consume(low_matches, low)

        for di, det in enumerate(high):
            if di in matched_high:
                continue
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    score=det.score,
                    age=0,
                    hits=1,
                    trail=[det.bbox.center()],
                    label=det.label,
                )
            )
            matched_ids.add(self._next_id)
            self._next_id += 1

        kept: list[Track] = []
        for track in self._tracks:
            if track.track_id not in matched_ids:
                track.age += 1
            if track.age <= self.max_age:
                kept.append(track)
        self._tracks = kept
        return [
            Track(
                track_id=t.track_id,
                bbox=t.bbox,
                score=t.score,
                age=t.age,
                hits=t.hits,
                trail=list(t.trail),
                label=t.label,
            )
            for t in self._tracks
            if t.hits >= self.min_hits and t.age == 0
        ]

    def _consume(self, matches: list[tuple[int, int]], detections: list[Detection]) -> None:
        for ti, di in matches:
            det = detections[di]
            track = self._tracks[ti]
            track.bbox = det.bbox
            track.score = det.score
            track.label = det.label
            track.age = 0
            track.hits += 1
            track.trail.append(det.bbox.center())
            if len(track.trail) > 90:
                del track.trail[:-90]
