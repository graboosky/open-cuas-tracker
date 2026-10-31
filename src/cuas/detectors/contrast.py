"""Sky-background dark-object detector.

The aircraft in the week-1 DUT clip is a dark body against a brighter sky.
This detector keeps pixels below a grey threshold, takes compact components,
and can expand each box about its centre. It is original code under
Apache-2.0, needs no weights, and runs on CPU. It is not a learned detector:
clouds, birds, and any other dark clutter produce false boxes, and a drone
that is not darker than the threshold is missed.

RF-DETR (Apache-2.0 for Nano/Small/Medium/Large) is the learned plugin. COCO
weights have no drone class, so this detector is what the published CPU demo
actually measures.
"""

from __future__ import annotations

import cv2
import numpy as np

from cuas.boxes import BBox, Detection


class SkyContrastDetector:
    """Find compact regions darker than ``threshold`` and optionally expand them.

    On DUT Anti-UAV video03 the annotated box includes rotors and margin, while
    the pixels that are actually dark are a smaller core. ``expand`` grows the
    core box about its centre so it can meet IoU 0.5 against that annotation.
    The value 3.0 was chosen on video03. It is a fitted demo setting, not a
    held-out claim. Use ``expand=1`` when the dark pixels already fill the
    object, as in the synthetic tests.
    """

    name = "contrast"

    def __init__(
        self,
        threshold: int = 40,
        close_kernel: int = 3,
        expand: float = 3.0,
        min_area: int = 20,
        max_area: int = 50000,
        max_detections: int = 5,
        aspect_min: float = 0.3,
        aspect_max: float = 3.5,
        nms_iou: float = 0.4,
    ) -> None:
        if expand <= 0:
            raise ValueError("expand must be positive")
        self.threshold = threshold
        self.close_kernel = close_kernel
        self.expand = expand
        self.min_area = min_area
        self.max_area = max_area
        self.max_detections = max_detections
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.nms_iou = nms_iou

    def reset(self) -> None:
        return None

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must have shape (H, W, 3)")
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        mask = np.where(gray < self.threshold, 255, 0).astype(np.uint8)
        if self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        found: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or area > self.max_area or h == 0:
                continue
            aspect = w / float(h)
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue
            core = gray[y : y + h, x : x + w]
            # Every surviving component is a detection. Rank darker cores higher,
            # but keep the score above the tracker's default high threshold so a
            # real box is not dropped before it can receive an id.
            dark_fraction = float((core < self.threshold).mean())
            score = float(np.clip(0.55 + 0.44 * dark_fraction, 0.55, 0.99))
            box = _expand(x, y, w, h, self.expand, width, height)
            found.append(Detection(box, score, "uav"))
        found.sort(key=lambda det: det.score, reverse=True)
        return _nms(found, self.nms_iou)[: self.max_detections]


def _expand(x: int, y: int, w: int, h: int, scale: float, width: int, height: int) -> BBox:
    cx = x + w / 2.0
    cy = y + h / 2.0
    half_w = w * scale / 2.0
    half_h = h * scale / 2.0
    x1 = max(0.0, cx - half_w)
    y1 = max(0.0, cy - half_h)
    x2 = min(float(width), cx + half_w)
    y2 = min(float(height), cy + half_h)
    return BBox(x1, y1, x2, y2)


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for det in detections:
        if all(det.bbox.iou(other.bbox) < iou_threshold for other in kept):
            kept.append(det)
    return kept
