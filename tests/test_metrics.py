from cuas.boxes import BBox, Detection
from cuas.metrics import score_detections


def _det(box: BBox, score: float = 0.9) -> Detection:
    return Detection(box, score, "uav")


def test_perfect_match_is_precision_recall_and_ap_one():
    gt = BBox(0, 0, 10, 10)
    scores = score_detections([[gt]], [[_det(gt)]])
    assert scores.true_positives == 1
    assert scores.false_positives == 0
    assert scores.false_negatives == 0
    assert scores.precision == 1
    assert scores.recall == 1
    assert scores.map50 == 1


def test_miss_and_false_positive():
    gt = BBox(0, 0, 10, 10)
    far = BBox(50, 50, 60, 60)
    scores = score_detections([[gt], [gt]], [[_det(far)], []])
    assert scores.true_positives == 0
    assert scores.false_positives == 1
    assert scores.false_negatives == 2
    assert scores.precision == 0
    assert scores.recall == 0
    assert scores.map50 == 0


def test_one_true_positive_and_one_false_positive():
    gt = BBox(0, 0, 10, 10)
    far = BBox(40, 40, 50, 50)
    scores = score_detections([[gt]], [[_det(gt, 0.9), _det(far, 0.2)]])
    assert scores.true_positives == 1
    assert scores.false_positives == 1
    assert scores.false_negatives == 0
    assert scores.precision == 0.5
    assert scores.recall == 1
    assert 0.5 < scores.map50 <= 1.0


def test_low_iou_is_not_a_match():
    gt = BBox(0, 0, 10, 10)
    shifted = BBox(6, 6, 16, 16)
    assert gt.iou(shifted) < 0.5
    scores = score_detections([[gt]], [[_det(shifted)]])
    assert scores.true_positives == 0
    assert scores.false_positives == 1


def test_empty_clip_with_no_objects():
    scores = score_detections([[]], [[]])
    assert scores.precision == 1
    assert scores.recall == 1
    assert scores.map50 == 0
