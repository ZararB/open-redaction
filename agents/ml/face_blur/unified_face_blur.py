"""Shared video/frame blurring pipeline with injectable face detectors."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from . import config as fb_config
from .boxes import box_iou, clamp_sane_face_box, nms_boxes, pad_box_for_redaction
from .byte_tracker import ByteFaceTracker
from .detectors import FaceDetector
from .tracking import FaceBoxTracker

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


class UnifiedFaceBlurService:
    def __init__(self, detector: FaceDetector, blur_strength: int = 51, confidence_threshold: float = 0.5):
        self.detector = detector
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.confidence_threshold = confidence_threshold

    def _apply_mask_to_roi(self, roi: np.ndarray, mask: np.ndarray) -> None:
        if fb_config.blur_style() == "blackout":
            roi[mask > 0] = 0
            return
        k = self.blur_strength
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        roi[:] = (blurred * m + roi * (1.0 - m)).astype(np.uint8)

    def _redaction_mask(self, w: int, h: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if fb_config.redaction_mask() == "ellipse":
            cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, min(w, h) // 16))
        else:
            mask[:] = 255
        return mask

    def _merge_detection_and_tracked(
        self,
        tracked: Sequence[Box],
        scored: Sequence[tuple[Box, float]],
        *,
        frame_w: int,
        frame_h: int,
    ) -> List[Box]:
        """Add high-confidence detections missing from tracks (frame-0 recall), without runaway boxes."""
        merged: List[Box] = []
        for box in tracked:
            sane = clamp_sane_face_box(box, frame_w, frame_h)
            if sane is not None:
                merged.append(sane)
        min_score = max(fb_config.low_confidence_threshold(), fb_config.merge_detection_min_score())
        for box, score in scored:
            if score < min_score:
                continue
            sane = clamp_sane_face_box(box, frame_w, frame_h)
            if sane is None:
                continue
            if not any(box_iou(sane, existing) >= 0.30 for existing in merged):
                merged.append(sane)
        return nms_boxes(merged, iou_threshold=0.42)

    def _apply_boxes(self, frame: np.ndarray, boxes: Sequence[Box]) -> Tuple[np.ndarray, int]:
        blurred_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        applied = 0
        for box in boxes:
            sane = clamp_sane_face_box(box, frame_w, frame_h)
            if sane is None:
                continue
            x, y, w, h = pad_box_for_redaction(
                sane, frame_w, frame_h, pad_ratio=fb_config.redaction_pad_ratio()
            )
            if w <= 0 or h <= 0:
                continue
            roi = blurred_frame[y : y + h, x : x + w]
            mask = self._redaction_mask(w, h)
            self._apply_mask_to_roi(roi, mask)
            applied += 1
        return blurred_frame, applied

    def blur_faces_in_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        faces = self.detector.detect(frame)
        return self._apply_boxes(frame, faces)

    def _probe_has_audio(self, input_video_path: str) -> bool:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    input_video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return "audio" in (result.stdout or "")
        except Exception:
            return False

    def _start_ffmpeg_writer(self, *, width: int, height: int, fps: float, output_path: str) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        # A pipe that is never drained can block ffmpeg on long videos.
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _mux_audio(self, *, video_path: str, audio_source: str, output_path: str) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_source,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)

    def blur_faces_in_video(
        self,
        input_video_path: str,
        output_video_path: str,
        sample_rate: float = 1.0,
    ) -> dict:
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        detect_interval = fb_config.detect_interval()
        if sample_rate > 1.0:
            # Legacy arg: sample_rate was misused as "detections per second". Prefer env interval.
            cap_probe = cv2.VideoCapture(input_video_path)
            probe_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
            cap_probe.release()
            detect_interval = max(detect_interval, max(1, int(probe_fps / sample_rate)))

        logger.info(
            "Blurring faces in video: %s (detector=%s detect_interval=%s hold=%s)",
            input_video_path,
            self.detector.name,
            detect_interval,
            fb_config.track_hold_frames(),
        )

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_video = output_video_path.replace(".mp4", "_video.mp4")
        if fb_config.byte_track_enabled():
            tracker: FaceBoxTracker | ByteFaceTracker = ByteFaceTracker(
                hold_frames=fb_config.track_hold_frames(),
                low_score=fb_config.low_confidence_threshold(),
                frame_w=width,
                frame_h=height,
            )
        else:
            tracker = FaceBoxTracker(
                hold_frames=fb_config.track_hold_frames(),
                frame_w=width,
                frame_h=height,
            )
        ffmpeg_proc = self._start_ffmpeg_writer(width=width, height=height, fps=fps, output_path=temp_video)

        total_faces = 0
        detection_runs = 0
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % detect_interval == 0:
                    scored = self.detector.detect_scored(frame)
                    if isinstance(tracker, ByteFaceTracker):
                        tracked = tracker.update(scored)
                    else:
                        tracked = tracker.update([box for box, _score in scored])
                    boxes = self._merge_detection_and_tracked(
                        tracked, scored, frame_w=width, frame_h=height
                    )
                    detection_runs += 1
                else:
                    boxes = tracker.tick()

                blurred_frame, num_faces = self._apply_boxes(frame, boxes)
                total_faces += num_faces
                ffmpeg_proc.stdin.write(blurred_frame.tobytes())
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.debug(
                        "Processed %s/%s frames, %s face-box applications",
                        frame_count,
                        total_frames,
                        total_faces,
                    )
        finally:
            cap.release()
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            rc = ffmpeg_proc.wait(timeout=300)
            if rc != 0:
                raise RuntimeError(f"ffmpeg_encode_failed:{rc}")

        has_audio = self._probe_has_audio(input_video_path)
        if has_audio:
            final_with_audio = output_video_path.replace(".mp4", "_av.mp4")
            try:
                self._mux_audio(video_path=temp_video, audio_source=input_video_path, output_path=final_with_audio)
                Path(temp_video).unlink(missing_ok=True)
                Path(final_with_audio).rename(output_video_path)
            except Exception as e:
                logger.warning("Audio mux failed (%s); using video-only output", e)
                Path(temp_video).rename(output_video_path)
        else:
            Path(temp_video).rename(output_video_path)

        logger.info(
            "Face blurring complete: %s face-box applications in %s frames (%s detection runs)",
            total_faces,
            frame_count,
            detection_runs,
        )
        return {
            "input_path": input_video_path,
            "output_path": output_video_path,
            "total_faces_blurred": total_faces,
            "frames_processed": frame_count,
            "processed_frames": frame_count,
            "detection_runs": detection_runs,
            "total_frames": total_frames,
            "success": True,
            "detector": self.detector.name,
        }

    def blur_faces_in_video_fast(self, input_video_path: str, output_video_path: str) -> dict:
        return self.blur_faces_in_video(input_video_path, output_video_path, sample_rate=1.0)
