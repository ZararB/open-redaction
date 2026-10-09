"""Build face blur service from environment (opencv vs ONNX + manifest)."""
from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .detectors import (
    EnsembleFaceDetector,
    OnnxSsdFaceDetector,
    OpenCvCaffeFaceDetector,
    ScrfdFaceDetector,
    YuNetFaceDetector,
)
from .manifest import validate_onnx_against_manifest
from .unified_face_blur import UnifiedFaceBlurService

logger = logging.getLogger(__name__)


def build_face_blur_service(
    blur_strength: int = config.DEFAULT_BLUR_STRENGTH,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE,
) -> UnifiedFaceBlurService:
    backend = config.face_blur_backend()
    detector = build_detector(backend, confidence_threshold)
    return UnifiedFaceBlurService(detector, blur_strength=blur_strength, confidence_threshold=confidence_threshold)


def _build_yunet(confidence_threshold: float) -> YuNetFaceDetector | None:
    from agents.ml.face_blur.model_assets import ensure_yunet_model, yunet_model_path

    yunet_path = yunet_model_path()
    if not ensure_yunet_model():
        return None
    return YuNetFaceDetector(
        str(yunet_path),
        confidence_threshold=confidence_threshold,
        low_confidence_threshold=config.low_confidence_threshold(),
        expand_to_head=config.head_expand_enabled(),
    )


def _build_scrfd(confidence_threshold: float) -> ScrfdFaceDetector | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.warning("onnxruntime unavailable; skipping SCRFD detector")
        return None

    from agents.ml.face_blur.model_assets import ensure_scrfd_model, scrfd_model_path

    scrfd_path = scrfd_model_path()
    if not scrfd_path.is_file():
        try:
            ensure_scrfd_model()
        except Exception as e:
            logger.warning("SCRFD model download failed (%s)", e)
    if not scrfd_path.is_file():
        return None
    return ScrfdFaceDetector(
        str(scrfd_path),
        confidence_threshold=confidence_threshold,
        low_confidence_threshold=config.low_confidence_threshold(),
        expand_to_head=config.head_expand_enabled(),
    )


def build_detector(backend: str, confidence_threshold: float):
    detector_kind = config.face_blur_detector()
    detectors: list = []

    if detector_kind in {"scrfd", "auto", "max"}:
        scrfd = _build_scrfd(confidence_threshold)
        if scrfd is not None:
            detectors.append(scrfd)

    if detector_kind in {"scrfd", "yunet", "auto", "max"}:
        yunet = _build_yunet(confidence_threshold)
        if yunet is not None:
            detectors.append(yunet)

    if (
        config.ensemble_enabled()
        and config.ensemble_opencv_ssd_enabled()
        and detector_kind in {"scrfd", "yunet", "auto", "max"}
    ):
        detectors.append(OpenCvCaffeFaceDetector(confidence_threshold=max(0.50, confidence_threshold)))

    if detectors:
        if len(detectors) == 1:
            return detectors[0]
        return EnsembleFaceDetector(detectors)

    if backend == "onnx":
        path = config.onnx_model_path()
        if path and Path(path).is_file():
            manifest = config.manifest_path()
            if manifest and Path(manifest).is_file():
                err = validate_onnx_against_manifest(path, manifest)
                if err:
                    logger.error("Face-blur manifest validation failed: %s — falling back to opencv", err)
                else:
                    try:
                        return OnnxSsdFaceDetector(path, confidence_threshold=confidence_threshold)
                    except Exception as e:
                        logger.warning("ONNX face detector init failed (%s); falling back to opencv", e)
            else:
                try:
                    return OnnxSsdFaceDetector(path, confidence_threshold=confidence_threshold)
                except Exception as e:
                    logger.warning("ONNX face detector init failed (%s); falling back to opencv", e)

    if backend != "opencv":
        logger.warning("Unknown FACE_BLUR_BACKEND=%s; using opencv", backend)
    return OpenCvCaffeFaceDetector(confidence_threshold=confidence_threshold)
