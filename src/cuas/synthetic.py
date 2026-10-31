"""Tiny synthetic clip used by tests and by ``--synthetic``. Not a drone dataset."""

from __future__ import annotations

import numpy as np

from cuas.boxes import BBox


def moving_square_clip(
    n_frames: int = 12,
    size: int = 160,
    box: int = 28,
    background: int = 210,
    foreground: int = 12,
) -> tuple[list[np.ndarray], list[list[BBox]]]:
    frames: list[np.ndarray] = []
    ground_truth: list[list[BBox]] = []
    for index in range(n_frames):
        image = np.full((size, size, 3), background, dtype=np.uint8)
        x = 8 + index * 4
        y = 50
        image[y : y + box, x : x + box] = foreground
        frames.append(image)
        ground_truth.append([BBox(float(x), float(y), float(x + box), float(y + box))])
    return frames, ground_truth
