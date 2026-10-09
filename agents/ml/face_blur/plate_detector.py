"""License plate box detection for local privacy redaction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import config
from .detectors import FaceDetector
from .plate_model_assets import _verified, ensure_plate_model, plate_model_path
from .yolo_onnx import YoloOnnxDetector
from .lpd_yunet import LpdYuNetPlateDetector

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


class LicensePlateDetector(FaceDetector):
    """YOLO ONNX plate detector; implements FaceDetector for tracker reuse."""

    def __init__(
        self,
        *,
        confidence_threshold: float | None = None,
        model_path: str | None = None,
    ):
        self.confidence_threshold = confidence_threshold or config.plate_confidence_threshold()
        path = model_path or str(plate_model_path())
        if model_path and _verified(Path(model_path)):
            pass
        elif not ensure_plate_model():
            raise FileNotFoundError(
                "license_plate_onnx_missing: set PRIVACY_PLATE_ONNX_MODEL_PATH, or configure "
                "PRIVACY_PLATE_AUTO_DOWNLOAD=1, PRIVACY_PLATE_ONNX_MODEL_URL and PRIVACY_PLATE_ONNX_SHA256"
            )
        else:
            path = str(plate_model_path())
        class_ids = config.plate_class_ids()
        self._engine = YoloOnnxDetector(
            path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=config.plate_iou_threshold(),
            input_size=config.plate_input_size(),
            class_ids=class_ids,
            name="LicensePlateYoloOnnx",
        )

    def detect(self, frame: np.ndarray) -> List[Box]:
        return self._engine.detect(frame)

    def detect_scored(self, frame: np.ndarray):
        return self._engine.detect_scored(frame)

    @property
    def name(self) -> str:
        return self._engine.name


def build_plate_detector_if_configured() -> LicensePlateDetector | LpdYuNetPlateDetector | None:
    try:
        if not config.plate_onnx_model_path() and not config.plate_auto_download_enabled():
            return LpdYuNetPlateDetector(confidence_threshold=config.plate_confidence_threshold())
        return LicensePlateDetector()
    except Exception as exc:
        logger.warning("License plate detector unavailable: %s", exc)
        return None
