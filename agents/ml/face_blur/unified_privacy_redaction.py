"""Face + license plate video redaction with shared tracking and blur pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2

from . import config as fb_config
from .byte_tracker import ByteFaceTracker
from .detectors import FaceDetector
from .plate_detector import build_plate_detector_if_configured
from .tracking import FaceBoxTracker
from .boxes import pad_box_for_redaction
from .manifest import sha256_file
from .unified_face_blur import UnifiedFaceBlurService

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def _model_artifacts(detector: FaceDetector | None, target: str) -> list[dict]:
    if detector is None:
        return []
    members = getattr(detector, "_detectors", None)
    if members is not None:
        return [item for member in members for item in _model_artifacts(member, target)]
    path = getattr(detector, "_onnx_path", None) or getattr(detector, "model_path", None)
    if path is None:
        path = getattr(getattr(detector, "_engine", None), "model_path", None)
    if path is None:
        return [{"target": target, "detector": detector.name, "artifact_sha256": None}]
    return [{"target": target, "detector": detector.name, "artifact_sha256": sha256_file(path)}]


class UnifiedPrivacyRedactionService(UnifiedFaceBlurService):
    """Extends face blur with optional license-plate detection and dual trackers."""

    def __init__(
        self,
        face_detector: FaceDetector | None,
        *,
        plate_detector: FaceDetector | None = None,
        blur_strength: int = 51,
        confidence_threshold: float = 0.5,
        blur_faces: bool = True,
        blur_plates: bool = True,
    ):
        primary = face_detector or plate_detector
        if primary is None:
            raise ValueError("no_redaction_targets_selected")
        super().__init__(primary, blur_strength=blur_strength, confidence_threshold=confidence_threshold)
        self.face_detector = face_detector
        self.plate_detector = plate_detector
        self.blur_faces = blur_faces and face_detector is not None
        self.blur_plates = blur_plates and plate_detector is not None

    @classmethod
    def from_env(
        cls,
        *,
        blur_strength: int = fb_config.DEFAULT_BLUR_STRENGTH,
        confidence_threshold: float = fb_config.DEFAULT_CONFIDENCE,
        blur_faces: bool = True,
        blur_plates: bool = True,
    ) -> "UnifiedPrivacyRedactionService":
        from .factory import build_detector

        backend = fb_config.face_blur_backend()
        face_det = build_detector(backend, confidence_threshold) if blur_faces else None
        plate_det = build_plate_detector_if_configured() if blur_plates else None
        if not blur_faces and not blur_plates:
            raise ValueError("no_redaction_targets_selected")
        if blur_faces and face_det is None:
            raise RuntimeError("face_detector_unavailable")
        if blur_faces and not any(
            item["artifact_sha256"] for item in _model_artifacts(face_det, "face")
        ) and not getattr(face_det, "_net", None):
            raise RuntimeError("face_model_unavailable_or_unverified")
        if blur_plates and plate_det is None:
            raise RuntimeError("license_plate_detector_unavailable")
        return cls(
            face_det,
            plate_detector=plate_det,
            blur_strength=blur_strength,
            confidence_threshold=confidence_threshold,
            blur_faces=blur_faces and face_det is not None,
            blur_plates=blur_plates and plate_det is not None,
        )

    def _make_tracker(self, width: int, height: int) -> FaceBoxTracker | ByteFaceTracker:
        if fb_config.byte_track_enabled():
            return ByteFaceTracker(
                hold_frames=fb_config.track_hold_frames(),
                low_score=fb_config.low_confidence_threshold(),
                frame_w=width,
                frame_h=height,
            )
        return FaceBoxTracker(hold_frames=fb_config.track_hold_frames(), frame_w=width, frame_h=height)

    def _tick_trackers(
        self,
        *,
        face_tracker: FaceBoxTracker | ByteFaceTracker | None,
        plate_tracker: FaceBoxTracker | ByteFaceTracker | None,
        frame,
        width: int,
        height: int,
        run_detection: bool,
    ) -> tuple[List[Box], List[Box]]:
        face_boxes: List[Box] = []
        plate_boxes: List[Box] = []
        if self.blur_faces and face_tracker is not None and self.face_detector is not None:
            if run_detection:
                scored = self.face_detector.detect_scored(frame)
                if isinstance(face_tracker, ByteFaceTracker):
                    tracked = face_tracker.update(scored)
                else:
                    tracked = face_tracker.update([box for box, _ in scored])
                face_boxes.extend(
                    self._merge_detection_and_tracked(tracked, scored, frame_w=width, frame_h=height)
                )
            else:
                face_boxes.extend(face_tracker.tick())

        if self.blur_plates and self.plate_detector is not None:
            if run_detection:
                # Face trackers and face-box sanity checks reject normal wide plates.
                # Plate inference runs on every frame until a plate-specific tracker is validated.
                plate_boxes.extend(box for box, _score in self.plate_detector.detect_scored(frame))
        return face_boxes, plate_boxes

    def _apply_privacy_boxes(self, frame, face_boxes: List[Box], plate_boxes: List[Box]):
        result, faces_applied = self._apply_boxes(frame, face_boxes)
        height, width = frame.shape[:2]
        plates_applied = 0
        for x, y, bw, bh in plate_boxes:
            if bw <= 0 or bh <= 0:
                continue
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + bw), min(height, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            px, py, pw, ph = pad_box_for_redaction((x0, y0, x1 - x0, y1 - y0), width, height, pad_ratio=0.15)
            # Opaque rectangle: no feathered edge or recoverable plate characters.
            result[py:py + ph, px:px + pw] = 0
            plates_applied += 1
        return result, faces_applied, plates_applied

    def redact_video(
        self,
        input_video_path: str,
        output_video_path: str,
        sample_rate: float = 1.0,
    ) -> dict:
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        if Path(input_video_path).resolve() == Path(output_video_path).resolve():
            raise ValueError("input_and_output_must_differ")

        # Privacy redaction cannot safely skip frames based on a sampling setting.
        detect_interval = 1

        logger.info(
            "Privacy redaction: %s faces=%s plates=%s detect_interval=%s",
            input_video_path,
            self.blur_faces,
            self.blur_plates,
            detect_interval,
        )

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not (0 < fps <= 240):
            cap.release()
            raise ValueError("invalid_video_metadata")

        face_tracker = self._make_tracker(width, height) if self.blur_faces else None
        plate_tracker = None

        output = Path(output_video_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_video = str(output.with_name(output.stem + ".redacting.mp4"))
        ffmpeg_proc = self._start_ffmpeg_writer(width=width, height=height, fps=fps, output_path=temp_video)

        total_faces = 0
        total_plates = 0
        detection_runs = 0
        frame_count = 0

        failed = True
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                run_detection = frame_count % detect_interval == 0
                face_boxes, plate_boxes = self._tick_trackers(
                    face_tracker=face_tracker,
                    plate_tracker=plate_tracker,
                    frame=frame,
                    width=width,
                    height=height,
                    run_detection=run_detection,
                )
                if run_detection:
                    detection_runs += 1
                blurred_frame, faces_applied, plates_applied = self._apply_privacy_boxes(frame, face_boxes, plate_boxes)
                total_faces += faces_applied
                total_plates += plates_applied
                ffmpeg_proc.stdin.write(blurred_frame.tobytes())
                frame_count += 1
            if frame_count == 0:
                raise ValueError("video_contains_no_decodable_frames")
            failed = False
        finally:
            cap.release()
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            rc = ffmpeg_proc.wait(timeout=300)
            if rc != 0 or failed:
                Path(temp_video).unlink(missing_ok=True)
            if rc != 0 and not failed:
                raise RuntimeError(f"ffmpeg_encode_failed:{rc}")
        # Audio can carry names and other identifying details; pre-upload output is silent.
        Path(temp_video).replace(output)

        targets: List[str] = []
        if self.blur_faces:
            targets.append("face")
        if self.blur_plates:
            targets.append("license-plate")

        return {
            "input_path": input_video_path,
            "output_path": output_video_path,
            "total_faces_blurred": total_faces,
            "total_plates_redacted": total_plates,
            "total_boxes_redacted": total_faces + total_plates,
            "frames_processed": frame_count,
            "processed_frames": frame_count,
            "detection_runs": detection_runs,
            "total_frames": total_frames,
            "success": True,
            "detector": self.face_detector.name if self.face_detector else "none",
            "plate_detector": self.plate_detector.name if self.plate_detector else None,
            "model_artifacts": _model_artifacts(self.face_detector, "face") + _model_artifacts(self.plate_detector, "license_plate"),
            "review_required": True,
            "targets": targets,
        }

    def redact_video_fast(self, input_video_path: str, output_video_path: str) -> dict:
        return self.redact_video(input_video_path, output_video_path, sample_rate=1.0)
