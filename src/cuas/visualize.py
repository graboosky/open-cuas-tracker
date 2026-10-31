"""Draw boxes, ids and trails, and a trajectory plot."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cuas.boxes import BBox, Track

_PALETTE = (
    (255, 176, 0),
    (0, 180, 255),
    (255, 80, 80),
    (180, 80, 255),
    (80, 220, 120),
    (255, 120, 200),
)


def annotate(
    frame_rgb: np.ndarray,
    tracks: list[Track],
    ground_truth: list[BBox] | None = None,
) -> np.ndarray:
    canvas = frame_rgb.copy()
    if ground_truth:
        for box in ground_truth:
            _rectangle(canvas, box, (40, 200, 40), 1)
        cv2.putText(
            canvas,
            "green = annotation",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 200, 40),
            1,
            cv2.LINE_AA,
        )
    for track in tracks:
        color = _PALETTE[(track.track_id - 1) % len(_PALETTE)]
        if len(track.trail) >= 2:
            points = np.array([[int(x), int(y)] for x, y in track.trail], dtype=np.int32)
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        _rectangle(canvas, track.bbox, color, 2)
        label = f"id {track.track_id} {track.score:.2f}"
        origin = (int(track.bbox.x1), max(16, int(track.bbox.y1) - 6))
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def plot_trajectory(
    ground_truth: list[list[BBox]],
    tracks_per_frame: list[list[Track]],
    path: str | Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    gt_x: list[float] = []
    gt_y: list[float] = []
    for boxes in ground_truth:
        for box in boxes:
            cx, cy = box.center()
            gt_x.append(cx)
            gt_y.append(cy)
    if gt_x:
        axis.plot(gt_x, gt_y, color="#2ca02c", linewidth=1.5, label="annotation center")
    series: dict[int, list[tuple[float, float]]] = {}
    for tracks in tracks_per_frame:
        for track in tracks:
            series.setdefault(track.track_id, []).append(track.bbox.center())
    for track_id, centers in series.items():
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        axis.plot(xs, ys, linewidth=1.2, label=f"track {track_id}")
    axis.set_title(title)
    axis.set_xlabel("x (px)")
    axis.set_ylabel("y (px)")
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    if gt_x or series:
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _rectangle(canvas: np.ndarray, box: BBox, color: tuple[int, int, int], thickness: int) -> None:
    cv2.rectangle(
        canvas,
        (int(round(box.x1)), int(round(box.y1))),
        (int(round(box.x2)), int(round(box.y2))),
        color,
        thickness,
        cv2.LINE_AA,
    )
