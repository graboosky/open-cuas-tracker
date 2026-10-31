"""Bounding boxes and detection records. Coordinates are pixels, origin top-left."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> BBox:
        return BBox(float(x), float(y), float(x) + float(w), float(y) + float(h))

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    label: str = "uav"


@dataclass
class Track:
    track_id: int
    bbox: BBox
    score: float
    age: int
    hits: int
    trail: list[tuple[float, float]]
    label: str = "uav"
