"""Download face-detector weights (shared by backend API and RunPod worker)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from . import config
from .manifest import sha256_file

logger = logging.getLogger(__name__)

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SCRFD_FILENAME = "scrfd_2.5g_bnkps.onnx"
SCRFD_URL = (
    "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/2.5g_bnkps.onnx"
)
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    logger.info("Downloading %s -> %s", url[:80], dest)
    urllib.request.urlretrieve(url, dest)


def opencv_ssd_dir() -> Path:
    return config.models_dir() / "opencv_ssd"


def yunet_model_path() -> Path:
    explicit = config.models_dir() / "yunet" / YUNET_FILENAME
    if explicit.is_file():
        return explicit
    return Path(__file__).resolve().parent / "models" / YUNET_FILENAME


def scrfd_model_path() -> Path:
    return config.models_dir() / "scrfd" / SCRFD_FILENAME


def ensure_scrfd_model() -> bool:
    try:
        _download(SCRFD_URL, scrfd_model_path())
        return scrfd_model_path().is_file()
    except Exception as exc:
        logger.warning("SCRFD model download failed: %s", exc)
        return False


def ensure_opencv_ssd_files() -> None:
    ssd_dir = opencv_ssd_dir()
    _download(PROTOTXT_URL, ssd_dir / "deploy.prototxt")
    _download(CAFFEMODEL_URL, ssd_dir / "res10_300x300_ssd_iter_140000.caffemodel")


def bundled_yunet_path() -> Path:
    return Path(__file__).resolve().parent / "models" / YUNET_FILENAME


def verify_bundled_yunet() -> Path:
    """Require shipped YuNet weights with pinned digest (no network)."""
    path = bundled_yunet_path()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("bundled_yunet_missing")
    if sha256_file(path) != YUNET_SHA256:
        raise ValueError("bundled_yunet_sha256_mismatch")
    return path


def verify_custom_onnx_face(path: Path) -> Path:
    """Custom face ONNX for worker/production must match a pinned digest."""
    import os

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("custom_face_onnx_missing")
    expected = os.environ.get("FACE_BLUR_ONNX_SHA256", "").strip().lower()
    if not expected:
        raise ValueError("face_onnx_sha256_required_for_custom")
    if sha256_file(path) != expected:
        raise ValueError("custom_face_onnx_sha256_mismatch")
    return path


def verify_local_scrfd() -> Path:
    """SCRFD is opt-in: file must already exist locally with pinned digest."""
    import os

    path = scrfd_model_path()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("scrfd_model_missing")
    expected = os.environ.get("FACE_BLUR_SCRFD_SHA256", "").strip().lower()
    if not expected:
        raise ValueError("scrfd_sha256_required_for_custom")
    if sha256_file(path) != expected:
        raise ValueError("scrfd_sha256_mismatch")
    return path


def ensure_yunet_model() -> bool:
    try:
        path = yunet_model_path()
        if path.is_file():
            if path.name == YUNET_FILENAME and path.parent.name == "models" and sha256_file(path) != YUNET_SHA256:
                raise ValueError("bundled_yunet_sha256_mismatch")
            return True
        _download(YUNET_URL, config.models_dir() / "yunet" / YUNET_FILENAME)
        return yunet_model_path().is_file()
    except Exception as exc:
        logger.warning("YuNet model download failed: %s", exc)
        return False
