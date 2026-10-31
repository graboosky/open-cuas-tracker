"""Write a small COCO detection tree for the optional RF-DETR fine-tune.

RF-DETR looks for ``train/_annotations.coco.json`` (and ``valid/``). Images
are written next to that file. This module does not download anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from cuas.boxes import BBox


def coco_dataset(images: list[dict]) -> dict:
    """Build a COCO dict.

    Each image dict needs ``file_name``, ``width``, ``height`` and ``boxes``
    as a list of :class:`BBox` in pixel corners (xyxy).
    """
    categories = [{"id": 0, "name": "uav", "supercategory": "aircraft"}]
    out_images = []
    annotations = []
    ann_id = 1
    for image_id, image in enumerate(images, start=1):
        out_images.append(
            {
                "id": image_id,
                "file_name": image["file_name"],
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
        for box in image["boxes"]:
            if not isinstance(box, BBox):
                raise TypeError("boxes must be BBox instances in xyxy pixels")
            w = box.width()
            h = box.height()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [box.x1, box.y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {"description": "DUT Anti-UAV subset exported by open-cuas-tracker"},
        "licenses": [{"id": 1, "name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"}],
        "categories": categories,
        "images": out_images,
        "annotations": annotations,
    }


def write_coco_split(images: list[dict], directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "_annotations.coco.json"
    path.write_text(json.dumps(coco_dataset(images)), encoding="utf-8")
    return path


def rows_from_detection_table(table) -> list[dict]:
    """Read an HF DUT detection parquet table into export rows.

    Expected columns match the ``aegean-ai/dut-anti-uav`` detection card:
    ``image_id``, ``image`` (struct with ``bytes``), ``width``, ``height``,
    and ``objects`` with ``bbox`` as ``[xmin, ymin, xmax, ymax]``.
    """
    rows: list[dict] = []
    for index in range(table.num_rows):
        image = table.column("image")[index].as_py()
        objects = table.column("objects")[index].as_py() or {}
        boxes: list[BBox] = []
        for raw in objects.get("bbox") or []:
            x1, y1, x2, y2 = (float(v) for v in raw)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(BBox(x1, y1, x2, y2))
        image_id = table.column("image_id")[index].as_py()
        blob = image["bytes"] if isinstance(image, dict) else image
        rows.append(
            {
                "file_name": f"{image_id}.jpg",
                "width": int(table.column("width")[index].as_py()),
                "height": int(table.column("height")[index].as_py()),
                "jpeg_bytes": blob,
                "boxes": boxes,
            }
        )
    return rows


def export_coco_tree(rows: list[dict], dest: str | Path, valid_fraction: float = 0.2) -> Path:
    """Write ``train/`` and ``valid/`` in the COCO layout RF-DETR expects.

    JPEG bytes are stored next to ``_annotations.coco.json``. ``dest`` should
    be a user directory or Google Drive, not the git tree.
    """
    if not rows:
        raise ValueError("no rows to export")
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between 0 and 0.5")
    dest = Path(dest)
    split_at = max(1, int(round(len(rows) * (1.0 - valid_fraction))))
    if split_at >= len(rows):
        split_at = len(rows) - 1
    for name, subset in (("train", rows[:split_at]), ("valid", rows[split_at:])):
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        for row in subset:
            (folder / row["file_name"]).write_bytes(row["jpeg_bytes"])
            images.append(
                {
                    "file_name": row["file_name"],
                    "width": row["width"],
                    "height": row["height"],
                    "boxes": row["boxes"],
                }
            )
        write_coco_split(images, folder)
    return dest
