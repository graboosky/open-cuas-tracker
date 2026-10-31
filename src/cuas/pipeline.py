"""Run detection, tracking, scoring and drawing on one clip."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cuas.boxes import BBox, Detection, Track
from cuas.detectors.base import Detector
from cuas.metrics import DetectionScores, score_detections
from cuas.tracker import IoUTracker
from cuas.visualize import annotate, plot_trajectory


@dataclass
class PipelineResult:
    detections: list[list[Detection]]
    tracks: list[list[Track]]
    annotated_frames: list[np.ndarray]
    scores: DetectionScores
    detector_fps: float
    end_to_end_fps: float
    detector_seconds: float
    end_to_end_seconds: float
    n_frames: int
    extra: dict = field(default_factory=dict)


def run_pipeline(
    frames_rgb: list[np.ndarray],
    ground_truth: list[list[BBox]],
    detector: Detector,
    tracker: IoUTracker | None = None,
) -> PipelineResult:
    if len(frames_rgb) != len(ground_truth):
        raise ValueError("frames and ground truth must have the same length")
    if tracker is None:
        tracker = IoUTracker()
    detector.reset()
    tracker.reset()
    detections: list[list[Detection]] = []
    tracks: list[list[Track]] = []
    annotated: list[np.ndarray] = []
    detector_seconds = 0.0
    end_to_end_seconds = 0.0
    for frame, gt in zip(frames_rgb, ground_truth):
        started = time.perf_counter()
        found = detector.detect(frame)
        after_detect = time.perf_counter()
        active = tracker.update(found)
        finished = time.perf_counter()
        detector_seconds += after_detect - started
        end_to_end_seconds += finished - started
        detections.append(found)
        tracks.append(active)
        annotated.append(annotate(frame, active, gt))
    n_frames = len(frames_rgb)
    scores = score_detections(ground_truth, detections, iou_threshold=0.5)
    return PipelineResult(
        detections=detections,
        tracks=tracks,
        annotated_frames=annotated,
        scores=scores,
        detector_fps=_fps(n_frames, detector_seconds),
        end_to_end_fps=_fps(n_frames, end_to_end_seconds),
        detector_seconds=detector_seconds,
        end_to_end_seconds=end_to_end_seconds,
        n_frames=n_frames,
    )


def save_trajectory(result: PipelineResult, ground_truth: list[list[BBox]], path: str | Path, title: str) -> None:
    plot_trajectory(ground_truth, result.tracks, path, title)


def _fps(n_frames: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return n_frames / seconds
