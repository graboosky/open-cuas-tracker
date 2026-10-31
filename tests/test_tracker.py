from cuas.boxes import BBox, Detection
from cuas.tracker import IoUTracker


def test_single_object_keeps_one_id():
    tracker = IoUTracker()
    ids = []
    for i in range(8):
        box = BBox(i * 3, 10, i * 3 + 20, 30)
        tracks = tracker.update([Detection(box, 0.9, "uav")])
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
        assert len(tracks[0].trail) == i + 1
    assert ids == [1] * 8


def test_two_objects_get_distinct_ids():
    tracker = IoUTracker()
    left = Detection(BBox(0, 0, 10, 10), 0.8, "uav")
    right = Detection(BBox(40, 0, 50, 10), 0.8, "uav")
    first = tracker.update([left, right])
    assert {t.track_id for t in first} == {1, 2}
    second = tracker.update(
        [
            Detection(BBox(2, 0, 12, 10), 0.8, "uav"),
            Detection(BBox(42, 0, 52, 10), 0.8, "uav"),
        ]
    )
    assert {t.track_id for t in second} == {1, 2}


def test_low_score_detection_does_not_start_a_track_but_can_continue_one():
    tracker = IoUTracker(high_thresh=0.5, low_thresh=0.1)
    started = tracker.update([Detection(BBox(0, 0, 10, 10), 0.9, "uav")])
    assert len(started) == 1
    continued = tracker.update([Detection(BBox(1, 0, 11, 10), 0.2, "uav")])
    assert len(continued) == 1
    assert continued[0].track_id == started[0].track_id
    tracker.reset()
    assert tracker.update([Detection(BBox(0, 0, 10, 10), 0.2, "uav")]) == []
