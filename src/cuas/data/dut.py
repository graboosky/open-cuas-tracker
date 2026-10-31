"""DUT Anti-UAV tracking clips from the public Hugging Face mirror.

The authors' repository is Apache-2.0:
https://github.com/wangdongdut/DUT-Anti-UAV
The mirror used here repeats that licence and is not gated:
https://huggingface.co/datasets/aegean-ai/dut-anti-uav

Only the requested video parquet is downloaded, into the Hugging Face cache.
Frames are not written into the git tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from cuas.boxes import BBox

HF_REPO = "aegean-ai/dut-anti-uav"
# Ground-truth marker used when the aircraft is outside the frame.
ABSENT = -100.0


@dataclass(frozen=True)
class TrackingClip:
    video_id: str
    frames_rgb: list[np.ndarray]
    ground_truth: list[list[BBox]]
    width: int
    height: int
    source: str

    @property
    def n_frames(self) -> int:
        return len(self.frames_rgb)


def parse_dut_bbox(values: list[float] | tuple[float, ...]) -> BBox | None:
    """DUT tracking rows are ``[x, y, w, h]``. Absent targets are all ``-100``."""
    if len(values) != 4:
        raise ValueError(f"expected 4 numbers, got {values!r}")
    x, y, w, h = (float(v) for v in values)
    if w <= 0 or h <= 0 or x <= ABSENT / 2 or y <= ABSENT / 2:
        return None
    return BBox.from_xywh(x, y, w, h)


def load_tracking_clip(video_id: str = "video03", cache_dir: str | Path | None = None) -> TrackingClip:
    """Download one tracking parquet if needed and decode it."""
    if not video_id.startswith("video") or not video_id[5:].isdigit():
        raise ValueError(f"unexpected DUT video id {video_id!r}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"tracking-parquet/{video_id}.parquet",
        repo_type="dataset",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    table = pq.read_table(path)
    frames: list[np.ndarray] = []
    ground_truth: list[list[BBox]] = []
    width = int(table.column("width")[0].as_py())
    height = int(table.column("height")[0].as_py())
    for index in range(table.num_rows):
        blob = table.column("image")[index].as_py()["bytes"]
        bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to decode frame {index} of {video_id}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        box = parse_dut_bbox(table.column("bbox")[index].as_py())
        ground_truth.append([] if box is None else [box])
    return TrackingClip(
        video_id=video_id,
        frames_rgb=frames,
        ground_truth=ground_truth,
        width=width,
        height=height,
        source=f"hf://datasets/{HF_REPO}/tracking-parquet/{video_id}.parquet",
    )
