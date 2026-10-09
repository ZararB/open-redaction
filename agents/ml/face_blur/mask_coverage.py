"""Opaque-mask coverage checks on encoded redaction output."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def _region_is_opaque(frame: np.ndarray, box: Box, *, max_pixel: int = 8) -> bool:
    x, y, w, h = box
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return True
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return True
    return bool(np.max(region) <= max_pixel)


def uncovered_annotation_boxes(frame: np.ndarray, boxes: Sequence[Box]) -> list[Box]:
    return [box for box in boxes if not _region_is_opaque(frame, box)]


def small_object_misses(boxes: Sequence[Box], uncovered: Sequence[Box], *, area_threshold: int = 400) -> int:
    uncovered_set = set(uncovered)
    return sum(1 for box in boxes if box in uncovered_set and box[2] * box[3] < area_threshold)


def decode_video_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"unreadable_encoded_video:{path}")
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"missing_encoded_frame:{path}:{frame_index}")
        return frame
    finally:
        cap.release()


def write_single_frame_video(frame: np.ndarray, dest: Path, *, fps: float = 10.0) -> None:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(dest),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError("video_writer_failed")
    writer.write(frame)
    writer.release()


def evaluate_encoded_frame_masks(
    service,
    frame: np.ndarray,
    *,
    face_boxes: Sequence[Box],
    plate_boxes: Sequence[Box],
    tmp_dir: Path,
) -> dict:
    """Run production encode path on a one-frame clip and verify GT regions are opaque."""
    source = tmp_dir / "mask_src.mp4"
    output = tmp_dir / "mask_out.mp4"
    write_single_frame_video(frame, source)
    result = service.redact_video_fast(str(source), str(output))
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "encoded_redaction_failed")
    encoded = decode_video_frame(output)
    face_miss = uncovered_annotation_boxes(encoded, face_boxes)
    plate_miss = uncovered_annotation_boxes(encoded, plate_boxes)
    return {
        "face_misses": face_miss,
        "plate_misses": plate_miss,
        "encoded_frames": 1,
    }


def summarize_mask_coverage(rows: Iterable[dict]) -> dict:
    first_frame_misses = 0
    small_misses = 0
    false_positive_pixels = 0
    temporal_gaps = 0
    examples: list[dict] = []
    frames_checked = 0
    for row in rows:
        frames_checked += 1
        if row.get("frame_index", 0) == 0 and (row.get("face_misses") or row.get("plate_misses")):
            first_frame_misses += 1
        small_misses += int(row.get("small_object_misses") or 0)
        false_positive_pixels += int(row.get("false_positive_pixels") or 0)
        temporal_gaps += int(row.get("temporal_gaps") or 0)
        if (row.get("face_misses") or row.get("plate_misses")) and len(examples) < 20:
            examples.append(
                {
                    "image": row.get("image"),
                    "face_misses": row.get("face_misses"),
                    "plate_misses": row.get("plate_misses"),
                }
            )
    total_misses = sum(
        len(row.get("face_misses") or []) + len(row.get("plate_misses") or [])
        for row in rows
    )
    passed = total_misses == 0 and temporal_gaps == 0
    return {
        "frames_checked": frames_checked,
        "annotation_misses": total_misses,
        "first_frame_misses": first_frame_misses,
        "small_object_misses": small_misses,
        "false_positive_pixels": false_positive_pixels,
        "temporal_gaps": temporal_gaps,
        "passed": passed,
        "examples": examples,
    }
