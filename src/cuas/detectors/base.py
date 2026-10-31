"""Detector interface. Implementations return per-frame boxes. They do not track."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from cuas.boxes import Detection


class Detector(Protocol):
    name: str

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Detect flying objects in one RGB frame (H, W, 3) uint8."""

    def reset(self) -> None:
        """Drop any frame-to-frame state."""
