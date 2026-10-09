from __future__ import annotations

import numpy as np

from agents.ml.face_blur.mask_coverage import _region_is_opaque, uncovered_annotation_boxes


def test_uncovered_boxes_detect_visible_pixels():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert uncovered_annotation_boxes(frame, [(10, 10, 20, 20)]) == [(10, 10, 20, 20)]
    frame[10:30, 10:30] = 0
    assert uncovered_annotation_boxes(frame, [(10, 10, 20, 20)]) == []


def test_region_is_opaque_allows_near_black():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[0, 0] = 5
    assert _region_is_opaque(frame, (0, 0, 2, 2))
