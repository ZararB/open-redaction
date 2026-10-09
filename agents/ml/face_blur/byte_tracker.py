"""ByteTrack-inspired two-stage association with Kalman prediction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .boxes import box_iou, clamp_sane_face_box, constrain_peak_box

Box = Tuple[int, int, int, int]
ScoredBox = Tuple[Box, float]


def _center_wh(box: Box) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0, float(w), float(h))


def _wh_to_xywh(cx: float, cy: float, w: float, h: float) -> Box:
    return (int(cx - w / 2), int(cy - h / 2), max(1, int(w)), max(1, int(h)))


@dataclass
class _KalmanBox:
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    frame_w: int = 1
    frame_h: int = 1

    def peek_predicted_box(self) -> Box:
        return _wh_to_xywh(self.cx + self.vx, self.cy + self.vy, self.w, self.h)

    def advance(self) -> None:
        self.cx += self.vx
        self.cy += self.vy
        self.cx = max(self.w / 2, min(self.cx, self.frame_w - self.w / 2))
        self.cy = max(self.h / 2, min(self.cy, self.frame_h - self.h / 2))

    def update(self, box: Box, *, alpha: float = 0.55) -> None:
        ncx, ncy, nw, nh = _center_wh(box)
        self.vx = 0.6 * self.vx + 0.4 * (ncx - self.cx)
        self.vy = 0.6 * self.vy + 0.4 * (ncy - self.cy)
        self.vx = max(-self.w * 0.35, min(self.vx, self.w * 0.35))
        self.vy = max(-self.h * 0.35, min(self.vy, self.h * 0.35))
        self.cx = (1 - alpha) * self.cx + alpha * ncx
        self.cy = (1 - alpha) * self.cy + alpha * ncy
        self.w = (1 - alpha) * self.w + alpha * nw
        self.h = (1 - alpha) * self.h + alpha * nh


@dataclass
class _Track:
    kalman: _KalmanBox
    peak_box: Box
    missed: int = 0

    def observe(self, box: Box) -> None:
        self.kalman.update(box)
        self.peak_box = constrain_peak_box(self.peak_box, box, self.kalman.frame_w, self.kalman.frame_h)
        self.missed = 0

    def match_box(self) -> Box:
        """Stable box for IoU matching (no side effects)."""
        return _wh_to_xywh(self.kalman.cx, self.kalman.cy, self.kalman.w, self.kalman.h)

    def redaction_box(self) -> Box:
        current = self.kalman.peek_predicted_box() if self.missed > 0 else _wh_to_xywh(
            self.kalman.cx, self.kalman.cy, self.kalman.w, self.kalman.h
        )
        merged = constrain_peak_box(self.peak_box, current, self.kalman.frame_w, self.kalman.frame_h)
        sane = clamp_sane_face_box(merged, self.kalman.frame_w, self.kalman.frame_h)
        return sane or merged


@dataclass
class ByteFaceTracker:
    """Two-threshold association (ByteTrack-style) + bounded peak-box retention."""

    hold_frames: int = 24
    high_match_iou: float = 0.30
    low_match_iou: float = 0.20
    high_score: float = 0.30
    low_score: float = 0.12
    frame_w: int = 1
    frame_h: int = 1
    _tracks: List[_Track] = field(default_factory=list)

    def _sanitize(self, box: Box) -> Box | None:
        return clamp_sane_face_box(box, self.frame_w, self.frame_h)

    def _greedy_match(
        self,
        detections: Sequence[Box],
        *,
        min_iou: float,
        allowed_tracks: set[int] | None = None,
    ) -> tuple[dict[int, int], set[int]]:
        matched: dict[int, int] = {}
        used_dets: set[int] = set()
        for det_idx, det in enumerate(detections):
            best_track = -1
            best_iou = min_iou
            for t_idx, track in enumerate(self._tracks):
                if allowed_tracks is not None and t_idx not in allowed_tracks:
                    continue
                if t_idx in matched:
                    continue
                score = box_iou(track.match_box(), det)
                if score >= best_iou:
                    best_iou = score
                    best_track = t_idx
            if best_track >= 0:
                matched[best_track] = det_idx
                used_dets.add(det_idx)
        return matched, used_dets

    def _spawn(self, box: Box) -> None:
        cx, cy, w, h = _center_wh(box)
        kalman = _KalmanBox(cx, cy, w, h, frame_w=self.frame_w, frame_h=self.frame_h)
        self._tracks.append(_Track(kalman=kalman, peak_box=box, missed=0))

    def update(self, scored: Sequence[ScoredBox]) -> List[Box]:
        high_dets: List[Box] = []
        low_dets: List[Box] = []
        for box, score in scored:
            sane = self._sanitize(box)
            if sane is None:
                continue
            if score >= self.high_score:
                high_dets.append(sane)
            elif score >= self.low_score:
                low_dets.append(sane)

        matched_high, used_high = self._greedy_match(high_dets, min_iou=self.high_match_iou)
        for t_idx, d_idx in matched_high.items():
            self._tracks[t_idx].observe(high_dets[d_idx])

        for d_idx, det in enumerate(high_dets):
            if d_idx in used_high:
                continue
            if not any(box_iou(t.match_box(), det) >= self.high_match_iou for t in self._tracks):
                self._spawn(det)

        unmatched_tracks = {i for i in range(len(self._tracks)) if i not in matched_high}
        matched_low, _ = self._greedy_match(
            low_dets,
            min_iou=self.low_match_iou,
            allowed_tracks=unmatched_tracks,
        )
        for t_idx, d_idx in matched_low.items():
            self._tracks[t_idx].observe(low_dets[d_idx])

        used_low = set(matched_low.values())
        for d_idx, det in enumerate(low_dets):
            if d_idx in used_low:
                continue
            if not any(box_iou(t.match_box(), det) >= self.low_match_iou for t in self._tracks):
                self._spawn(det)

        for t_idx in unmatched_tracks:
            if t_idx not in matched_low:
                self._tracks[t_idx].missed += 1

        self._tracks = [t for t in self._tracks if t.missed <= self.hold_frames]
        return [t.redaction_box() for t in self._tracks]

    def tick(self) -> List[Box]:
        for track in self._tracks:
            track.missed += 1
            if track.missed > 0:
                track.kalman.advance()
        self._tracks = [t for t in self._tracks if t.missed <= self.hold_frames]
        return [t.redaction_box() for t in self._tracks]

    def active_boxes(self) -> List[Box]:
        return [t.redaction_box() for t in self._tracks]
