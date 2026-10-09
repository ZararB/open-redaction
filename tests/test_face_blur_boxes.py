import numpy as np

from agents.ml.face_blur.boxes import clamp_sane_face_box, expand_face_to_head_box, nms_boxes


def test_head_expansion_covers_forehead():
    landmarks = np.array(
        [
            [50, 50],
            [70, 50],
            [60, 65],
            [52, 78],
            [68, 78],
        ],
        dtype=np.float32,
    )
    box = expand_face_to_head_box(
        x=48,
        y=48,
        w=24,
        h=32,
        landmarks=landmarks,
        frame_w=200,
        frame_h=200,
    )
    x, y, w, h = box
    assert y < 48
    assert h > 32


def test_clamp_shrinks_full_frame_box():
    huge = (0, 0, 640, 480)
    sane = clamp_sane_face_box(huge, 640, 480)
    assert sane is not None
    _x, _y, w, h = sane
    assert w * h < 640 * 480 * 0.12


def test_nms_deduplicates_overlapping_boxes():
    boxes = [(0, 0, 100, 100), (10, 10, 90, 90), (200, 200, 40, 40)]
    kept = nms_boxes(boxes, iou_threshold=0.4)
    assert len(kept) == 2
