"""Single-class detection scores at a fixed IoU threshold.

Precision is TP / (TP + FP). When the detector emits nothing, precision is 0
if there is at least one ground-truth box, and 1 if there is nothing to find.
Recall is TP / (TP + FN), or 1 when there are no ground-truth boxes.
mAP@0.5 is the all-point interpolated average precision of the score-ranked
detections (VOC 2010 style) for one class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cuas.boxes import BBox, Detection


@dataclass(frozen=True)
class DetectionScores:
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    n_ground_truth: int
    n_detections: int
    precision: float
    recall: float
    map50: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _greedy_match(
    gts: list[BBox],
    dets: list[Detection],
    iou_threshold: float,
    taken_gt: set[int] | None = None,
) -> list[tuple[int, int]]:
    """Match detections to ground truth by descending IoU. Each box is used once."""
    blocked = taken_gt if taken_gt is not None else set()
    pairs: list[tuple[float, int, int]] = []
    for di, det in enumerate(dets):
        for gi, gt in enumerate(gts):
            if gi in blocked:
                continue
            overlap = det.bbox.iou(gt)
            if overlap >= iou_threshold:
                pairs.append((overlap, di, gi))
    pairs.sort(reverse=True)
    used_d: set[int] = set()
    used_g: set[int] = set(blocked)
    matches: list[tuple[int, int]] = []
    for _overlap, di, gi in pairs:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        matches.append((di, gi))
    return matches


def average_precision(
    ground_truth: list[list[BBox]],
    detections: list[list[Detection]],
    iou_threshold: float = 0.5,
) -> float:
    """All-point interpolated AP at one IoU threshold, one class."""
    if len(ground_truth) != len(detections):
        raise ValueError("ground_truth and detections must have one entry per frame")
    n_gt = sum(len(frame) for frame in ground_truth)
    ranked: list[tuple[float, int, int]] = []
    for frame_index, frame_dets in enumerate(detections):
        for det_index, det in enumerate(frame_dets):
            ranked.append((float(det.score), frame_index, det_index))
    if n_gt == 0 or not ranked:
        return 0.0
    ranked.sort(reverse=True)
    matched: list[set[int]] = [set() for _ in ground_truth]
    tp: list[int] = []
    fp: list[int] = []
    for _score, frame_index, det_index in ranked:
        det = detections[frame_index][det_index]
        best_iou = 0.0
        best_gt = -1
        for gt_index, gt in enumerate(ground_truth[frame_index]):
            if gt_index in matched[frame_index]:
                continue
            overlap = det.bbox.iou(gt)
            if overlap > best_iou:
                best_iou = overlap
                best_gt = gt_index
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched[frame_index].add(best_gt)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def score_detections(
    ground_truth: list[list[BBox]],
    detections: list[list[Detection]],
    iou_threshold: float = 0.5,
) -> DetectionScores:
    """Precision, recall and mAP at ``iou_threshold`` for one class."""
    if len(ground_truth) != len(detections):
        raise ValueError("ground_truth and detections must have one entry per frame")
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    n_gt = 0
    n_det = 0
    for gts, dets in zip(ground_truth, detections):
        n_gt += len(gts)
        n_det += len(dets)
        matches = _greedy_match(gts, dets, iou_threshold)
        true_positives += len(matches)
        false_positives += len(dets) - len(matches)
        false_negatives += len(gts) - len(matches)
    if true_positives + false_positives == 0:
        precision = 1.0 if n_gt == 0 else 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
    if n_gt == 0:
        recall = 1.0
    else:
        recall = true_positives / n_gt
    map50 = average_precision(ground_truth, detections, iou_threshold)
    return DetectionScores(
        iou_threshold=iou_threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        n_ground_truth=n_gt,
        n_detections=n_det,
        precision=precision,
        recall=recall,
        map50=map50,
    )
