"""OpenCV Zoo LPD_YuNet plate detector with bounded, checked output decoding.

Model: OpenCV Zoo license_plate_detection_lpd_yunet_2023mar.onnx.
The upstream model and its license are under models/; implementation follows the
published loc/conf/iou output contract without copying the upstream Python code.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import cv2
import numpy as np

from .manifest import sha256_file

MODEL_NAME = "license_plate_detection_lpd_yunet_2023mar.onnx"
MODEL_SHA256 = "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7"


def bundled_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / MODEL_NAME


class LpdYuNetPlateDetector:
    name = "OpenCVZooLPD-YuNet-2023mar"
    input_width = 320
    input_height = 240

    def __init__(self, model_path: str | Path | None = None, *, confidence_threshold: float = 0.35):
        self.model_path = str(model_path or bundled_model_path())
        if sha256_file(self.model_path) != MODEL_SHA256:
            raise ValueError("bundled_plate_model_sha256_mismatch")
        self.confidence_threshold = confidence_threshold
        self._net = cv2.dnn.readNetFromONNX(self.model_path)
        self._priors = self._build_priors()

    @classmethod
    def _build_priors(cls) -> np.ndarray:
        w, h = cls.input_width, cls.input_height
        fm2 = (int((h + 1) / 2) // 2, int((w + 1) / 2) // 2)
        maps = []
        current = fm2
        for _ in range(4):
            current = (current[0] // 2, current[1] // 2)
            maps.append(current)
        sizes = ((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))
        steps = (8, 16, 32, 64)
        priors = [
            ((j + 0.5) * steps[k] / w, (i + 0.5) * steps[k] / h, size / w, size / h)
            for k, fmap in enumerate(maps)
            for i, j in product(range(fmap[0]), range(fmap[1]))
            for size in sizes[k]
        ]
        return np.asarray(priors, dtype=np.float32)

    def detect_scored(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("empty_plate_detection_frame")
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        self._net.setInput(cv2.dnn.blobFromImage(resized))
        loc, conf, iou = self._net.forward(["loc", "conf", "iou"])
        count = len(self._priors)
        if loc.shape != (count, 14) or conf.shape != (count, 2) or iou.shape != (count, 1):
            raise ValueError(f"unexpected_lpd_yunet_output:{loc.shape},{conf.shape},{iou.shape}")
        if not all(np.isfinite(array).all() for array in (loc, conf, iou)):
            raise ValueError("nonfinite_lpd_yunet_output")
        scores = np.sqrt(np.maximum(0.0, conf[:, 1] * np.clip(iou[:, 0], 0, 1)))
        selected = np.flatnonzero(scores >= self.confidence_threshold)
        if not len(selected):
            return []
        # Four quadrilateral corners are encoded at offsets 4:8 and 10:14.
        offsets = loc[selected][:, [4, 5, 6, 7, 10, 11, 12, 13]].reshape(-1, 4, 2)
        prior = self._priors[selected]
        points = (prior[:, None, 0:2] + offsets * 0.1 * prior[:, None, 2:4])
        points *= np.array([width, height], dtype=np.float32)
        x0 = np.clip(points[:, :, 0].min(axis=1), 0, width)
        y0 = np.clip(points[:, :, 1].min(axis=1), 0, height)
        x1 = np.clip(points[:, :, 0].max(axis=1), 0, width)
        y1 = np.clip(points[:, :, 1].max(axis=1), 0, height)
        boxes = np.column_stack((x0, y0, x1 - x0, y1 - y0)).astype(np.int32)
        valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        boxes = boxes[valid]
        selected = selected[valid]
        if not len(selected):
            return []
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores[selected].tolist(), self.confidence_threshold, 0.3)
        return [(tuple(int(v) for v in boxes[int(i)]), float(scores[selected[int(i)]])) for i in np.asarray(keep).reshape(-1)]

    def detect(self, frame: np.ndarray):
        return [box for box, _ in self.detect_scored(frame)]
