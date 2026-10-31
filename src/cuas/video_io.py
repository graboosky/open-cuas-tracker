"""Read and write short RGB clips. Dataset files are never committed."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def write_mp4(frames_rgb: list[np.ndarray], path: str | Path, fps: float = 10.0) -> None:
    if not frames_rgb:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {path}")
    try:
        for frame in frames_rgb:
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError("all frames must share the same size")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def read_mp4(path: str | Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def write_gif(
    frames_rgb: list[np.ndarray],
    path: str | Path,
    fps: float = 10.0,
    max_width: int = 640,
) -> None:
    if not frames_rgb:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    for frame in frames_rgb:
        image = Image.fromarray(frame)
        if image.width > max_width:
            height = max(1, int(round(image.height * max_width / image.width)))
            image = image.resize((max_width, height), Image.Resampling.BILINEAR)
        images.append(image)
    duration_ms = max(1, int(round(1000.0 / fps)))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
