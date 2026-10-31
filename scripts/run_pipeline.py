#!/usr/bin/env python3
"""Run the detection and tracking demo locally.

Examples
--------
Measured demo (downloads DUT Anti-UAV video03, about 2 MB, Apache-2.0):

    python scripts/run_pipeline.py --video-id video03 --output runs/video03

Offline synthetic clip, no download:

    python scripts/run_pipeline.py --synthetic --output runs/synthetic
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuas import __version__
from cuas.data.dut import HF_REPO, load_tracking_clip
from cuas.detectors import build_detector
from cuas.pipeline import run_pipeline, save_trajectory
from cuas.synthetic import moving_square_clip
from cuas.tracker import IoUTracker
from cuas.video_io import write_gif, write_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Open counter-UAS detection and tracking demo.")
    parser.add_argument("--video-id", default="video03", help="DUT tracking video id, video01..video20.")
    parser.add_argument("--synthetic", action="store_true", help="Use a generated square instead of DUT.")
    parser.add_argument("--detector", default="contrast", choices=["contrast", "rfdetr"])
    parser.add_argument("--rfdetr-size", default="nano", help="Apache-2.0 size: nano, small, medium, large.")
    parser.add_argument("--rfdetr-weights", default=None, help="Optional fine-tuned checkpoint. Not downloaded by default.")
    parser.add_argument("--max-frames", type=int, default=0, help="Use only the first N frames. 0 means the whole clip.")
    parser.add_argument("--output", type=Path, default=Path("runs/video03"))
    parser.add_argument("--gif", type=Path, default=None, help="Extra GIF path. The output directory always gets preview.gif.")
    parser.add_argument("--gif-width", type=int, default=480, help="Maximum GIF width in pixels.")
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args()

    if args.synthetic:
        frames, ground_truth = moving_square_clip()
        source = "synthetic moving square (not a drone dataset)"
        clip_id = "synthetic"
        width = frames[0].shape[1]
        height = frames[0].shape[0]
    else:
        clip = load_tracking_clip(args.video_id)
        frames, ground_truth = clip.frames_rgb, clip.ground_truth
        source = clip.source
        clip_id = clip.video_id
        width, height = clip.width, clip.height

    if args.max_frames > 0:
        frames = frames[: args.max_frames]
        ground_truth = ground_truth[: args.max_frames]

    detector_kwargs = {}
    if args.detector == "contrast" and args.synthetic:
        # The synthetic square is uniformly dark, so it must not be expanded.
        detector_kwargs = {"expand": 1.0}
    if args.detector == "rfdetr":
        detector_kwargs = {"size": args.rfdetr_size, "weights": args.rfdetr_weights}
    detector = build_detector(args.detector, **detector_kwargs)
    result = run_pipeline(frames, ground_truth, detector, IoUTracker())

    args.output.mkdir(parents=True, exist_ok=True)
    gif_path = args.output / "preview.gif"
    mp4_path = args.output / "annotated.mp4"
    traj_path = args.trajectory or (args.output / "trajectory.png")
    metrics_path = args.metrics or (args.output / "metrics.json")
    write_gif(result.annotated_frames, gif_path, fps=8.0, max_width=args.gif_width)
    write_mp4(result.annotated_frames, mp4_path, fps=10.0)
    title = f"{clip_id} / {detector.name}"
    save_trajectory(result, ground_truth, traj_path, title)
    if args.gif is not None:
        write_gif(result.annotated_frames, args.gif, fps=8.0, max_width=args.gif_width)
    if args.trajectory is not None and args.trajectory != traj_path:
        save_trajectory(result, ground_truth, args.trajectory, title)

    payload = {
        "package_version": __version__,
        "clip_id": clip_id,
        "source": source,
        "dataset_repo": None if args.synthetic else HF_REPO,
        "licence_note": (
            "synthetic"
            if args.synthetic
            else "DUT Anti-UAV via the Apache-2.0 Hugging Face mirror aegean-ai/dut-anti-uav"
        ),
        "detector": detector.name,
        "tracker": "IoUTracker (Apache-2.0, this repository; ByteTrack-style two-stage IoU)",
        "width": width,
        "height": height,
        "n_frames": result.n_frames,
        "playback_fps_of_gif": 8,
        "detector_settings": {
            "threshold": getattr(detector, "threshold", None),
            "expand": getattr(detector, "expand", None),
            "close_kernel": getattr(detector, "close_kernel", None),
            "size": getattr(detector, "size", None),
        },
        "tuning_note": (
            "SkyContrastDetector defaults (threshold 40, expand 3, close kernel 3) "
            "were chosen on video03. Other clips, including video02, use those same "
            "settings and are not a held-out success."
        ),
        "capture_fps": None,
        "capture_fps_note": "DUT does not state the camera frame rate. FPS below is processing speed.",
        "host": platform.platform(),
        "processor": platform.processor(),
        **result.scores.as_dict(),
        "detector_fps": result.detector_fps,
        "end_to_end_fps": result.end_to_end_fps,
        "detector_seconds": result.detector_seconds,
        "end_to_end_seconds": result.end_to_end_seconds,
        "track_frames": {
            str(track_id): count
            for track_id, count in sorted(Counter(
                track.track_id for frame_tracks in result.tracks for track in frame_tracks
            ).items())
        },
        "gif": str(gif_path),
        "video": str(mp4_path),
        "trajectory": str(traj_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
