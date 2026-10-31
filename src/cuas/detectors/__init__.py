"""Detector registry. The learned default is RF-DETR Nano (Apache-2.0).

``contrast`` is the weight-free detector used for the measured CPU demo.
"""

from __future__ import annotations

from cuas.detectors.base import Detector
from cuas.detectors.contrast import SkyContrastDetector
from cuas.detectors.rfdetr import RFDetrDetector

__all__ = ["Detector", "RFDetrDetector", "SkyContrastDetector", "build_detector"]


def build_detector(name: str, **kwargs) -> Detector:
    key = name.strip().lower()
    if key in {"contrast", "sky", "motion"}:
        return SkyContrastDetector(**kwargs)
    if key in {"rfdetr", "rf-detr"}:
        return RFDetrDetector(**kwargs)
    if key in {"yolov5", "yolov8", "yolo11", "yolov11", "ultralytics"}:
        raise ValueError(
            "Ultralytics YOLO is AGPL-3.0 and is not the default. "
            "Use --detector rfdetr (Apache-2.0 for Nano/Small/Medium/Large) "
            "or --detector contrast. An AGPL model can be added behind the "
            "Detector protocol for research, in a separate process, if you "
            "accept the AGPL obligations."
        )
    raise ValueError(f"Unknown detector {name!r}. Choose contrast or rfdetr.")
