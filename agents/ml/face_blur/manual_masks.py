"""Apply admin-drawn redaction masks on top of AI-processed video."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def _norm_mask_to_box(mask: Dict[str, Any], frame_w: int, frame_h: int) -> Box | None:
    try:
        x = float(mask.get("x", 0))
        y = float(mask.get("y", 0))
        w = float(mask.get("w", 0))
        h = float(mask.get("h", 0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    x0 = max(0, int(x * frame_w))
    y0 = max(0, int(y * frame_h))
    x1 = min(frame_w, int((x + w) * frame_w))
    y1 = min(frame_h, int((y + h) * frame_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _mask_active(mask: Dict[str, Any], time_sec: float) -> bool:
    if not bool(mask.get("enabled", True)):
        return False
    try:
        start = float(mask.get("start_sec") or 0.0)
    except (TypeError, ValueError):
        start = 0.0
    if time_sec < start:
        return False
    end_raw = mask.get("end_sec")
    if end_raw is None or end_raw == "":
        return True
    try:
        end = float(end_raw)
    except (TypeError, ValueError):
        return True
    return time_sec <= end


def _shape_mask(w: int, h: int, shape: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if str(shape).lower() == "rect":
        mask[:] = 255
        return mask
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, min(w, h) // 16))
    return mask


def _apply_box_blackout(frame: np.ndarray, box: Box, *, shape: str) -> None:
    x, y, w, h = box
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return
    mask = _shape_mask(w, h, shape)
    roi[mask > 0] = 0


def apply_manual_masks_to_video(
    *,
    input_video_path: str,
    output_video_path: str,
    masks: Sequence[Dict[str, Any]],
    blur_style: str = "blackout",
) -> Dict[str, Any]:
    """Burn manual masks into video (typically on top of AI output)."""
    if not Path(input_video_path).exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    enabled = [m for m in masks if isinstance(m, dict)]
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video = output_video_path.replace(".mp4", "_manual_video.mp4")
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
        temp_video,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_count = 0
    applications = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            time_sec = frame_count / fps if fps > 0 else 0.0
            for mask in enabled:
                if not _mask_active(mask, time_sec):
                    continue
                box = _norm_mask_to_box(mask, width, height)
                if not box:
                    continue
                shape = str(mask.get("shape") or "ellipse")
                if blur_style == "blackout":
                    _apply_box_blackout(frame, box, shape=shape)
                else:
                    x, y, w, h = box
                    roi = frame[y : y + h, x : x + w]
                    if roi.size:
                        k = 51 if 51 % 2 == 1 else 53
                        blurred = cv2.GaussianBlur(roi, (k, k), 0)
                        m = (_shape_mask(w, h, shape).astype(np.float32) / 255.0)[..., None]
                        roi[:] = (blurred * m + roi * (1.0 - m)).astype(np.uint8)
                applications += 1
            proc.stdin.write(frame.tobytes())
            frame_count += 1
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait(timeout=600)
        if rc != 0:
            raise RuntimeError(f"ffmpeg_manual_encode_failed: {stderr[-2000:]}")

    # Mux audio from source when present.
    try:
        probe = subprocess.run(
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
        if "audio" in (probe.stdout or ""):
            av_out = output_video_path.replace(".mp4", "_manual_av.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_video,
                    "-i",
                    input_video_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    av_out,
                ],
                capture_output=True,
                timeout=300,
                check=True,
            )
            Path(temp_video).unlink(missing_ok=True)
            Path(av_out).rename(output_video_path)
        else:
            Path(temp_video).rename(output_video_path)
    except Exception as exc:
        logger.warning("Manual redaction audio mux failed (%s); using video-only", exc)
        Path(temp_video).rename(output_video_path)

    logger.info(
        "Manual mask apply complete: %s applications across %s frames",
        applications,
        frame_count,
    )
    return {
        "success": True,
        "frames_processed": frame_count,
        "total_frames": total_frames,
        "mask_applications": applications,
        "masks_applied": len(enabled),
    }
