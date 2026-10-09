"""Temporal face-box tracking to reduce flicker and occlusion leaks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .boxes import box_union, clamp_sane_face_box, constrain_peak_box

Box = Tuple[int, int, int, int]


def _iou(a: Box, b: Box) -> float:
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


def _smooth_box(prev: Box, new: Box, alpha: float = 0.45) -> Box:
    px, py, pw, ph = prev
    nx, ny, nw, nh = new
    x = int(px * (1 - alpha) + nx * alpha)
    y = int(py * (1 - alpha) + ny * alpha)
    w = int(pw * (1 - alpha) + nw * alpha)
    h = int(ph * (1 - alpha) + nh * alpha)
    return (x, y, max(1, w), max(1, h))


def _center(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


@dataclass
class _Track:
    box: Box
    peak_box: Box
    missed: int = 0
    vx: float = 0.0
    vy: float = 0.0
    frame_w: int = 1
    frame_h: int = 1

    def observe(self, detection: Box, *, smooth_alpha: float) -> None:
        cx0, cy0 = _center(self.box)
        self.box = _smooth_box(self.box, detection, smooth_alpha)
        self.peak_box = constrain_peak_box(self.peak_box, detection, self.frame_w, self.frame_h)
        cx1, cy1 = _center(self.box)
        self.vx = 0.65 * self.vx + 0.35 * (cx1 - cx0)
        self.vy = 0.65 * self.vy + 0.35 * (cy1 - cy0)
        self.missed = 0

    def predict_box(self) -> Box:
        x, y, w, h = self.box
        px = int(x + self.vx)
        py = int(y + self.vy)
        return (px, py, w, h)

    def redaction_box(self) -> Box:
        current = self.box if self.missed == 0 else self.predict_box()
        merged = constrain_peak_box(self.peak_box, current, self.frame_w, self.frame_h)
        return clamp_sane_face_box(merged, self.frame_w, self.frame_h) or merged


@dataclass
class FaceBoxTracker:
    """IoU tracker with velocity prediction and peak-box retention."""

    hold_frames: int = 18
    match_iou: float = 0.18
    smooth_alpha: float = 0.45
    frame_w: int = 1
    frame_h: int = 1
    _tracks: List[_Track] = field(default_factory=list)

    def update(self, detections: Sequence[Box]) -> List[Box]:
        matched_track_idx: set[int] = set()
        for det in detections:
            sane = clamp_sane_face_box(det, self.frame_w, self.frame_h)
            if sane is None:
                continue
            best_idx = -1
            best_iou = self.match_iou
            for idx, track in enumerate(self._tracks):
                if idx in matched_track_idx:
                    continue
                score = _iou(track.box, sane)
                if score >= best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx >= 0:
                self._tracks[best_idx].observe(sane, smooth_alpha=self.smooth_alpha)
                matched_track_idx.add(best_idx)
            else:
                self._tracks.append(
                    _Track(box=sane, peak_box=sane, missed=0, frame_w=self.frame_w, frame_h=self.frame_h)
                )
                matched_track_idx.add(len(self._tracks) - 1)

        for idx, track in enumerate(self._tracks):
            if idx not in matched_track_idx:
                track.missed += 1

        self._tracks = [t for t in self._tracks if t.missed <= self.hold_frames]
        return [t.redaction_box() for t in self._tracks]

    def tick(self) -> List[Box]:
        """Advance missed counter on frames without fresh detections."""
        for track in self._tracks:
            track.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.hold_frames]
        return [t.redaction_box() for t in self._tracks]

    def active_boxes(self) -> List[Box]:
        return [t.redaction_box() for t in self._tracks]
