from agents.ml.face_blur.byte_tracker import ByteFaceTracker

_FRAME = dict(frame_w=640, frame_h=480)


def test_byte_tracker_spawns_from_low_confidence_on_cold_start():
    tracker = ByteFaceTracker(hold_frames=5, high_score=0.5, low_score=0.1, **_FRAME)
    low_only = ((50, 50, 40, 40), 0.25)
    boxes = tracker.update([low_only])
    assert len(boxes) == 1


def test_byte_tracker_keeps_peak_box_on_low_confidence_recovery():
    tracker = ByteFaceTracker(hold_frames=5, high_score=0.5, low_score=0.1, **_FRAME)
    big = ((100, 80, 120, 120), 0.9)
    tracker.update([big])
    partial = ((130, 110, 40, 40), 0.2)
    boxes = tracker.update([partial])
    assert boxes[0][2] >= 40


def test_byte_tracker_holds_through_empty_frames():
    # hold_frames must cover the initial missed bump from an unmatched spawn in update()
    tracker = ByteFaceTracker(hold_frames=4, high_score=0.4, low_score=0.1, **_FRAME)
    tracker.update([((10, 10, 50, 50), 0.8)])
    for _ in range(3):
        assert len(tracker.tick()) == 1
    assert tracker.tick() == []
