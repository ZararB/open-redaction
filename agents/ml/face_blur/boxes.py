"""Face box geometry helpers: head expansion, NMS, union."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


def box_area(box: Box) -> int:
    _x, _y, w, h = box
    return max(0, w) * max(0, h)


def box_union(a: Box, b: Box) -> Box:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes: Sequence[Box], iou_threshold: float = 0.45) -> List[Box]:
    if not boxes:
        return []
    arr = sorted(boxes, key=box_area, reverse=True)
    kept: List[Box] = []
    for candidate in arr:
        if all(box_iou(candidate, k) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def expand_face_to_head_box(
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    landmarks: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    forehead_ratio: float = 0.55,
    chin_ratio: float = 0.30,
    side_ratio: float = 0.28,
) -> Box:
    """
    Expand a tight face bbox to cover forehead/hair and jaw — critical when
    microphones or hands partially occlude the lower face.
    """
    if landmarks is not None and landmarks.shape[0] >= 5:
        xs = np.concatenate([landmarks[:, 0], [x, x + w]])
        ys = np.concatenate([landmarks[:, 1], [y, y + h]])
        lx, rx = float(xs.min()), float(xs.max())
        ty, by = float(ys.min()), float(ys.max())
        # Nose + mouth landmarks: extend below visible face for mic/hand occlusion.
        nose = landmarks[2]
        mouth_y = float(max(landmarks[3][1], landmarks[4][1]))
        eye_y = float(min(landmarks[0][1], landmarks[1][1]))
        mid_face = max(1.0, mouth_y - eye_y)
        mic_pad = max(chin_ratio * (by - ty), 0.42 * mid_face)
        by = max(by, mouth_y + mic_pad)
        ty = min(ty, eye_y - forehead_ratio * mid_face * 1.15)
        _ = nose  # nose anchors mid-face geometry when mouth is occluded
    else:
        lx, ty, rx, by = float(x), float(y), float(x + w), float(y + h)

    face_w = max(1.0, rx - lx)
    face_h = max(1.0, by - ty)
    lx = max(0.0, lx - side_ratio * face_w)
    rx = min(float(frame_w), rx + side_ratio * face_w)
    ty = max(0.0, ty - forehead_ratio * face_h)
    by = min(float(frame_h), by + chin_ratio * face_h)
    return (int(lx), int(ty), max(1, int(rx - lx)), max(1, int(by - ty)))


MAX_FACE_BOX_WIDTH_FRAC = 0.24
MAX_FACE_BOX_HEIGHT_FRAC = 0.30
MAX_FACE_BOX_AREA_FRAC = 0.085
PEAK_GROWTH_RATIO = 1.12
PEAK_GROWTH_MAX_RATIO = 1.18


def clamp_sane_face_box(box: Box, frame_w: int, frame_h: int) -> Box | None:
    """Drop or shrink boxes that are implausible for a single face."""
    x, y, w, h = box
    if w <= 0 or h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    ratio = w / float(h)
    if ratio > 2.1 or ratio < 0.48:
        return None
    max_w = max(8, int(frame_w * MAX_FACE_BOX_WIDTH_FRAC))
    max_h = max(8, int(frame_h * MAX_FACE_BOX_HEIGHT_FRAC))
    max_area = int(frame_w * frame_h * MAX_FACE_BOX_AREA_FRAC)
    cx, cy = x + w / 2.0, y + h / 2.0
    if w > max_w or h > max_h:
        w = min(w, max_w)
        h = min(h, max_h)
    if w * h > max_area:
        scale = (max_area / float(w * h)) ** 0.5
        w = max(8, int(w * scale))
        h = max(8, int(h * scale))
    x = int(max(0, min(cx - w / 2.0, frame_w - w)))
    y = int(max(0, min(cy - h / 2.0, frame_h - h)))
    if w * h < 64:
        return None
    return (int(x), int(y), int(w), int(h))


def constrain_peak_box(peak: Box, latest: Box, frame_w: int, frame_h: int) -> Box:
    """Hold a little extra for occlusion without unbounded union growth."""
    latest_s = clamp_sane_face_box(latest, frame_w, frame_h)
    peak_s = clamp_sane_face_box(peak, frame_w, frame_h)
    if latest_s is None:
        return peak_s or peak
    lx, ly, lw, lh = latest_s
    cx, cy = lx + lw / 2.0, ly + lh / 2.0
    grow_w = min(lw * PEAK_GROWTH_RATIO, frame_w * MAX_FACE_BOX_WIDTH_FRAC)
    grow_h = min(lh * PEAK_GROWTH_RATIO, frame_h * MAX_FACE_BOX_HEIGHT_FRAC)
    if peak_s:
        _px, _py, pw, ph = peak_s
        # A partial face detection can be much smaller than the last complete
        # head. Retain prior coverage while bounding growth by the frame.
        grow_w = min(max(grow_w, pw), max(lw, pw) * PEAK_GROWTH_MAX_RATIO, frame_w * MAX_FACE_BOX_WIDTH_FRAC)
        grow_h = min(max(grow_h, ph), max(lh, ph) * PEAK_GROWTH_MAX_RATIO, frame_h * MAX_FACE_BOX_HEIGHT_FRAC)
    grow_w = max(8, int(grow_w))
    grow_h = max(8, int(grow_h))
    x = int(max(0, min(cx - grow_w / 2.0, frame_w - grow_w)))
    y = int(max(0, min(cy - grow_h / 2.0, frame_h - grow_h)))
    return clamp_sane_face_box((x, y, grow_w, grow_h), frame_w, frame_h) or (x, y, grow_w, grow_h)


def pad_box_for_redaction(box: Box, frame_w: int, frame_h: int, pad_ratio: float = 0.05) -> Box:
    x, y, w, h = box
    pad = max(6, int(pad_ratio * max(w, h)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame_w, x + w + pad)
    y1 = min(frame_h, y + h + pad)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
