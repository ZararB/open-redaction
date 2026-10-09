from agents.ml.face_blur.tracking import FaceBoxTracker, _iou


def test_iou_overlap():
    assert _iou((0, 0, 100, 100), (50, 50, 100, 100)) > 0.1
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_tracker_holds_box_when_detection_drops():
    tracker = FaceBoxTracker(hold_frames=3, frame_w=200, frame_h=200)
    box = (10, 10, 40, 40)
    redaction_box = tracker.update([box])
    assert redaction_box[0][0] <= box[0]
    assert redaction_box[0][2] >= box[2]
    assert tracker.update([]) == redaction_box
    assert tracker.update([]) == redaction_box
    assert tracker.update([]) == redaction_box
    assert tracker.update([]) == []


def test_tracker_smooths_jitter():
    tracker = FaceBoxTracker(hold_frames=5, smooth_alpha=0.5, frame_w=200, frame_h=200)
    first = (10, 10, 40, 40)
    second = (12, 12, 40, 40)
    tracker.update([first])
    tracker.update([second])
    # The public redaction box deliberately retains peak coverage; inspect the
    # smoothed track state when verifying jitter behavior.
    smoothed = tracker._tracks[0].box
    assert smoothed[0] == (first[0] + second[0]) // 2
    assert smoothed[1] == (first[1] + second[1]) // 2


def test_peak_box_does_not_shrink_on_partial_detection():
    tracker = FaceBoxTracker(hold_frames=5, frame_w=500, frame_h=500)
    large = (0, 0, 100, 100)
    partial = (30, 30, 30, 30)
    tracker.update([large])
    boxes = tracker.update([partial])
    assert boxes[0][2] >= 100
    assert boxes[0][3] >= 100


def test_tick_advances_missed_counter():
    tracker = FaceBoxTracker(hold_frames=2, frame_w=200, frame_h=200)
    box = (0, 0, 50, 50)
    redaction_box = tracker.update([box])
    assert tracker.tick() == redaction_box
    assert tracker.tick() == redaction_box
    assert tracker.tick() == []
