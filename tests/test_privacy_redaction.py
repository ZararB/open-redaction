from __future__ import annotations

import numpy as np
import pytest
import cv2
import subprocess

from agents.ml.face_blur.evaluation import match_boxes
from agents.ml.face_blur.unified_privacy_redaction import UnifiedPrivacyRedactionService
from agents.ml.face_blur.yolo_onnx import YoloOnnxDetector
from agents.ml.face_blur.lpd_yunet import LpdYuNetPlateDetector, MODEL_SHA256, bundled_model_path
from agents.ml.face_blur.manifest import sha256_file
from agents.ml.face_blur.model_assets import YUNET_SHA256, yunet_model_path


class Detector:
    name = "fake"

    def __init__(self, boxes):
        self.boxes = boxes

    def detect_scored(self, frame):
        return [(box, 0.99) for box in self.boxes]

    def detect(self, frame):
        return self.boxes


def test_wide_plate_is_covered_without_face_box_sanitization():
    service = UnifiedPrivacyRedactionService(
        None, plate_detector=Detector([(30, 20, 100, 18)]), blur_faces=False, blur_plates=True
    )
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    faces, plates = service._tick_trackers(
        face_tracker=None, plate_tracker=None, frame=frame,
        width=200, height=100, run_detection=True,
    )
    result, face_count, plate_count = service._apply_privacy_boxes(frame, faces, plates)
    assert face_count == 0 and plate_count == 1
    assert np.all(result[20:38, 30:130] == 0)


def test_detector_error_propagates_instead_of_returning_unredacted_frame():
    class Broken(Detector):
        def detect_scored(self, frame):
            raise RuntimeError("detector_failed")

    service = UnifiedPrivacyRedactionService(None, plate_detector=Broken([]), blur_faces=False)
    with pytest.raises(RuntimeError, match="detector_failed"):
        service._tick_trackers(
            face_tracker=None, plate_tracker=None,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            width=10, height=10, run_detection=True,
        )


def test_annotation_matching_is_one_to_one():
    assert match_boxes([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 10)]) == (1, 1, 0)


def test_yolo_rejects_unsupported_output_contract():
    engine = object.__new__(YoloOnnxDetector)
    engine.session = type("Session", (), {"run": lambda self, *args: [np.zeros((1, 4, 10))]})()
    engine.input_name = "images"
    engine.input_size = 640
    with pytest.raises(ValueError, match="unsupported_plate_model_output"):
        engine.detect_scored(np.zeros((32, 32, 3), dtype=np.uint8))


def test_short_video_redacts_all_frames_and_emits_silent_mp4(tmp_path):
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 80))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((80, 160, 3), 255, dtype=np.uint8))
    writer.release()
    service = UnifiedPrivacyRedactionService(
        None, plate_detector=Detector([(20, 30, 110, 15)]), blur_faces=False, blur_plates=True
    )
    result = service.redact_video(str(source), str(output))
    assert result["processed_frames"] == 4
    assert result["total_plates_redacted"] == 4
    cap = cv2.VideoCapture(str(output))
    seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame[35:40, 40:100].mean() < 10
        seen += 1
    cap.release()
    assert seen == 4
    streams = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=index", "-of", "csv=p=0", str(output),
    ], text=True)
    assert not streams.strip()


def test_bundled_model_digests_and_plate_output_contract():
    assert sha256_file(bundled_model_path()) == MODEL_SHA256
    assert sha256_file(yunet_model_path()) == YUNET_SHA256
    detector = LpdYuNetPlateDetector()
    assert len(detector._priors) == 4385
    assert detector.detect_scored(np.zeros((240, 320, 3), dtype=np.uint8)) == []
