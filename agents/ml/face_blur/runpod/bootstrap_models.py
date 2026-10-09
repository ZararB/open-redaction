"""Offline model bootstrap for long-running workers (no cloud SDK)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agents.ml.face_blur.lpd_yunet import MODEL_SHA256 as PLATE_SHA256, bundled_model_path as bundled_plate_path
from agents.ml.face_blur.manifest import sha256_file
from agents.ml.face_blur.model_assets import (
    verify_bundled_yunet,
    verify_custom_onnx_face,
    verify_local_scrfd,
)

logger = logging.getLogger(__name__)


class BootstrapModelError(RuntimeError):
    """Raised when required models are missing or fail digest verification."""


def verify_bundled_plate_model() -> Path:
    path = bundled_plate_path()
    if not path.is_file() or path.stat().st_size == 0:
        raise BootstrapModelError("bundled_plate_missing")
    if sha256_file(path) != PLATE_SHA256:
        raise BootstrapModelError("bundled_plate_sha256_mismatch")
    return path


def _verify_custom_plate() -> None:
    raw = os.environ.get("PRIVACY_PLATE_ONNX_MODEL_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    expected = os.environ.get("PRIVACY_PLATE_ONNX_SHA256", "").strip().lower()
    if not expected:
        raise BootstrapModelError("sha256_required")
    if not path.is_file() or sha256_file(path) != expected:
        raise BootstrapModelError("mismatch")


def configure_face_blur_backend(requested: str = "ztiles") -> str:
    """
    Verify bundled YuNet + plate weights and optional custom ONNX paths.

    Default: bundled YuNet (offline), OpenCV backend string for compatibility.
    """
    del requested  # reserved for API compatibility with older worker entrypoints
    try:
        verify_bundled_yunet()
        verify_bundled_plate_model()
    except (FileNotFoundError, ValueError) as exc:
        raise BootstrapModelError(str(exc)) from exc

    detector = os.environ.get("FACE_BLUR_DETECTOR", "").strip().lower()
    if detector == "onnx":
        custom = os.environ.get("FACE_BLUR_ONNX_MODEL_PATH", "").strip()
        if not custom:
            raise BootstrapModelError("sha256")
        try:
            verify_custom_onnx_face(Path(custom))
        except ValueError as exc:
            msg = str(exc)
            if "sha256" in msg:
                raise BootstrapModelError(msg) from exc
            raise BootstrapModelError("sha256") from exc
        os.environ["FACE_BLUR_BACKEND"] = "onnx"
        return "onnx"

    if detector in {"scrfd", "max", "auto"}:
        scrfd_path = os.environ.get("FACE_BLUR_SCRFD_SHA256", "").strip()
        if scrfd_path:
            try:
                verify_local_scrfd()
            except (FileNotFoundError, ValueError) as exc:
                raise BootstrapModelError(str(exc)) from exc
            os.environ["FACE_BLUR_DETECTOR"] = "scrfd"
            os.environ.setdefault("FACE_BLUR_BACKEND", "onnx")
            return "scrfd"

    _verify_custom_plate()

    os.environ["FACE_BLUR_DETECTOR"] = "yunet"
    os.environ["FACE_BLUR_BACKEND"] = "opencv"
    logger.info("Open Redaction worker using bundled YuNet + LPD-YuNet")
    return "yunet"
