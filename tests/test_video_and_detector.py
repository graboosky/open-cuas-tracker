from pathlib import Path

import numpy as np

from cuas.data.coco import coco_dataset
from cuas.data.dut import parse_dut_bbox
from cuas.detectors import build_detector
from cuas.detectors.contrast import SkyContrastDetector
from cuas.pipeline import run_pipeline
from cuas.synthetic import moving_square_clip
from cuas.video_io import read_mp4, write_gif, write_mp4


def test_absent_dut_box_is_missing():
    assert parse_dut_bbox([-100, -100, -100, -100]) is None
    box = parse_dut_bbox([10, 20, 30, 40])
    assert box is not None
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 20, 40, 60)


def test_contrast_finds_dark_square_and_pipeline_recalls_it():
    frames, ground_truth = moving_square_clip()
    # expand=1: the synthetic square is dark all the way to its edges.
    # The DUT demo uses expand=3 because annotations include a brighter margin.
    result = run_pipeline(frames, ground_truth, SkyContrastDetector(expand=1.0))
    assert result.scores.recall >= 0.8
    assert result.scores.precision >= 0.8
    assert result.detector_fps > 0
    assert result.end_to_end_fps > 0
    assert result.annotated_frames[0].shape == frames[0].shape


def test_video_roundtrip(tmp_path: Path):
    frames, _gt = moving_square_clip(n_frames=4, size=64, box=12)
    mp4 = tmp_path / "clip.mp4"
    gif = tmp_path / "clip.gif"
    write_mp4(frames, mp4, fps=5)
    write_gif(frames, gif, fps=5, max_width=32)
    loaded = read_mp4(mp4)
    assert len(loaded) == 4
    assert loaded[0].shape == (64, 64, 3)
    assert gif.stat().st_size > 0
    assert abs(int(loaded[0].mean()) - int(frames[0].mean())) < 8


def test_detection_parquet_rows_export_coco(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cuas.data.coco import export_coco_tree, rows_from_detection_table

    image = pa.array(
        [{"bytes": b"jpeg-bytes", "path": None}, {"bytes": b"jpeg-2", "path": None}],
        type=pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
    )
    objects = pa.array(
        [
            {"bbox": [[1.0, 2.0, 11.0, 12.0]], "category": [0]},
            {"bbox": [[0.0, 0.0, 4.0, 5.0]], "category": [0]},
        ],
        type=pa.struct(
            [
                pa.field("bbox", pa.list_(pa.list_(pa.float32()))),
                pa.field("category", pa.list_(pa.int64())),
            ]
        ),
    )
    table = pa.table(
        {
            "image_id": ["00001", "00002"],
            "image": image,
            "width": [20, 20],
            "height": [20, 20],
            "objects": objects,
        }
    )
    path = tmp_path / "mini.parquet"
    pq.write_table(table, path)
    rows = rows_from_detection_table(pq.read_table(path))
    dest = export_coco_tree(rows, tmp_path / "coco")
    assert (dest / "train" / "00001.jpg").read_bytes() == b"jpeg-bytes"
    assert (dest / "valid" / "_annotations.coco.json").is_file()


def test_coco_export_uses_xywh_and_uav_category():
    from cuas.boxes import BBox

    payload = coco_dataset(
        [
            {
                "file_name": "a.jpg",
                "width": 100,
                "height": 80,
                "boxes": [BBox(1, 2, 11, 12)],
            }
        ]
    )
    assert payload["categories"][0]["name"] == "uav"
    assert payload["annotations"][0]["bbox"] == [1, 2, 10, 10]
    assert payload["annotations"][0]["category_id"] == 0


def test_agpl_and_pml_detectors_are_refused():
    import pytest

    with pytest.raises(ValueError, match="AGPL"):
        build_detector("yolov8")
    with pytest.raises(ValueError, match="Platform Model License"):
        build_detector("rfdetr", size="xl")


def test_gif_width_is_capped(tmp_path: Path):
    frame = np.full((100, 400, 3), 180, dtype=np.uint8)
    path = tmp_path / "wide.gif"
    write_gif([frame, frame], path, fps=5, max_width=80)
    from PIL import Image

    with Image.open(path) as image:
        assert image.size[0] == 80
