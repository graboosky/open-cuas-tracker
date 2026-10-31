"""RF-DETR plugin (Roboflow). Optional: imported only when this detector is built.

Licence, checked against the upstream README and LICENSE on 2026-10-31:

* Package code: Apache-2.0 (https://github.com/roboflow/rf-detr/blob/develop/LICENSE).
* Detection weights Nano, Small, Medium, Large: Apache-2.0.
* Detection weights Atto, Femto, Pico, XL, 2XL (``rfdetr[plus]``): Platform
  Model License 1.0, not Apache-2.0. This class refuses those sizes.

COCO-pretrained checkpoints do not have a drone class. Treat their recall on
DUT as unknown until you fine-tune. Ultralytics YOLO is intentionally not
wrapped: YOLOv5, YOLOv8 and YOLO11 are AGPL-3.0.
"""

from __future__ import annotations

import numpy as np

from cuas.boxes import BBox, Detection

APACHE_SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Documented by Roboflow as PML-1.0 detection checkpoints. Refused on purpose.
PML_SIZES = {"atto", "femto", "pico", "xl", "xlarge", "2xl", "2xlarge"}


class RFDetrDetector:
    name = "rfdetr"

    def __init__(
        self,
        size: str = "nano",
        threshold: float = 0.35,
        weights: str | None = None,
        allowed_labels: set[str] | None = None,
    ) -> None:
        key = size.strip().lower()
        if key in PML_SIZES:
            raise ValueError(
                f"RF-DETR size {size!r} is published under the Platform Model License 1.0, "
                "not Apache-2.0. Use nano, small, medium, or large."
            )
        if key not in APACHE_SIZES:
            raise ValueError(
                f"Unknown RF-DETR size {size!r}. Choose one of: {', '.join(sorted(APACHE_SIZES))}."
            )
        self.size = key
        self.threshold = threshold
        self.weights = weights
        self.allowed_labels = {label.lower() for label in allowed_labels} if allowed_labels else None
        self._model = None
        self._class_names: dict[int, str] = {}

    def reset(self) -> None:
        return None

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        model = self._load()
        from PIL import Image

        image = Image.fromarray(frame_rgb)
        raw = model.predict(image, threshold=self.threshold)
        xyxy = getattr(raw, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return []
        scores = getattr(raw, "confidence", None)
        class_ids = getattr(raw, "class_id", None)
        names = _names_from_detections(raw, self._class_names)
        found: list[Detection] = []
        for index, box in enumerate(xyxy):
            score = float(scores[index]) if scores is not None else 1.0
            class_id = int(class_ids[index]) if class_ids is not None else -1
            label = names.get(class_id, f"class_{class_id}")
            if self.allowed_labels is not None and label.lower() not in self.allowed_labels:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            found.append(Detection(BBox(x1, y1, x2, y2), score, label))
        return found

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import rfdetr
        except ImportError as exc:
            raise ImportError(
                "RF-DETR is optional. Install it with `pip install -r requirements-dl.txt` "
                "(Apache-2.0 sizes only; do not install rfdetr[plus])."
            ) from exc
        cls = getattr(rfdetr, APACHE_SIZES[self.size])
        kwargs = {}
        if self.weights:
            kwargs["pretrain_weights"] = self.weights
        self._model = cls(**kwargs)
        self._class_names = _coco_names(rfdetr)
        return self._model


def _coco_names(rfdetr_module) -> dict[int, str]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except ImportError:
        return {}
    if isinstance(COCO_CLASSES, dict):
        return {int(k): str(v) for k, v in COCO_CLASSES.items()}
    return {index: str(name) for index, name in enumerate(COCO_CLASSES)}


def _names_from_detections(raw, fallback: dict[int, str]) -> dict[int, str]:
    data = getattr(raw, "data", None) or {}
    class_names = data.get("class_name") if isinstance(data, dict) else None
    class_ids = getattr(raw, "class_id", None)
    if class_names is None or class_ids is None:
        return fallback
    names = dict(fallback)
    for class_id, class_name in zip(class_ids, class_names):
        names[int(class_id)] = str(class_name)
    return names
