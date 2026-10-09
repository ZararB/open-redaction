"""License-plate detector ONNX weights (optional auto-download)."""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from . import config
from .manifest import sha256_file

logger = logging.getLogger(__name__)

DEFAULT_PLATE_FILENAME = "yolov8n_license_plate.onnx"


def plate_model_path() -> Path:
    explicit = config.plate_onnx_model_path()
    if explicit:
        return Path(explicit)
    return config.models_dir() / "plates" / DEFAULT_PLATE_FILENAME


def plate_model_url() -> str:
    return config.plate_onnx_model_url() or ""


def _verified(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    expected = os.environ.get("PRIVACY_PLATE_ONNX_SHA256", "").strip().lower()
    if expected and sha256_file(path) != expected:
        raise ValueError("license_plate_model_sha256_mismatch")
    return True


def ensure_plate_model() -> bool:
    path = plate_model_path()
    if _verified(path):
        return True
    if not config.plate_auto_download_enabled():
        return False
    url = plate_model_url()
    if not url.startswith("https://"):
        raise ValueError("license_plate_model_https_url_required")
    if not os.environ.get("PRIVACY_PLATE_ONNX_SHA256", "").strip():
        raise ValueError("license_plate_model_sha256_required_for_download")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading license-plate ONNX model -> %s", path)
        fd, tmp_name = tempfile.mkstemp(prefix=".plate-model-", suffix=".onnx", dir=path.parent)
        os.close(fd)
        temp_path = Path(tmp_name)
        try:
            urllib.request.urlretrieve(url, temp_path)
            if not _verified(temp_path):
                raise ValueError("license_plate_model_empty")
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.warning("License-plate model download failed: %s", exc)
        return False
