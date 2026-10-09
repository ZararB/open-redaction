from agents.ml.face_blur.manual_masks import _mask_active, _norm_mask_to_box


def test_norm_mask_to_box():
    box = _norm_mask_to_box({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, 1000, 500)
    assert box == (100, 100, 300, 200)


def test_mask_active_respects_time_window():
    mask = {"enabled": True, "start_sec": 1.0, "end_sec": 3.0}
    assert not _mask_active(mask, 0.5)
    assert _mask_active(mask, 2.0)
    assert not _mask_active(mask, 4.0)


def test_mask_active_open_ended():
    mask = {"enabled": True, "start_sec": 0.0, "end_sec": None}
    assert _mask_active(mask, 99.0)
