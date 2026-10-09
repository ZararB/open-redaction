"""
Configuration for the Open Redaction face-detection stack.

Environment:
  FACE_BLUR_BACKEND
    - opencv (default): OpenCV DNN Caffe Res10_300x300 SSD + Haar fallback
    - onnx: ONNX Runtime with an SSD-compatible face detector (see onnx_detector.py)

  FACE_BLUR_ONNX_MODEL_PATH
    Path to .onnx weights when FACE_BLUR_BACKEND=onnx.

  FACE_BLUR_MODEL_MANIFEST_PATH
    Optional JSON manifest (version, sha256, input size) for ops validation.

  FACE_BLUR_MODEL_VERSION
    Optional string recorded in logs / manifest checks.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_BACKEND = "opencv"
DEFAULT_DETECTOR = "yunet"  # bundled, pinned OpenCV Zoo baseline; scrfd/auto opt in
DEFAULT_CONFIDENCE = 0.30
DEFAULT_LOW_CONFIDENCE = 0.16
DEFAULT_BLUR_STRENGTH = 51
DEFAULT_BLUR_STYLE = "blackout"  # blackout | gaussian
DEFAULT_REDACTION_MASK = "rect"  # opaque rectangular coverage is the privacy-safe default
DEFAULT_DETECT_INTERVAL = 1  # run detector every N frames (1 = all frames)
DEFAULT_TRACK_HOLD_FRAMES = 24  # keep boxes ~0.8s at 30fps when detection drops
DEFAULT_ENSEMBLE = True  # SCRFD + YuNet + OpenCV SSD merge for recall
DEFAULT_BYTE_TRACK = True  # ByteTrack-style two-threshold association


def face_blur_backend() -> str:
    return os.environ.get("FACE_BLUR_BACKEND", DEFAULT_BACKEND).strip().lower()


def face_blur_detector() -> str:
    return os.environ.get("FACE_BLUR_DETECTOR", DEFAULT_DETECTOR).strip().lower()


def blur_style() -> str:
    return os.environ.get("FACE_BLUR_STYLE", DEFAULT_BLUR_STYLE).strip().lower()


def detect_interval() -> int:
    raw = os.environ.get("FACE_BLUR_DETECT_INTERVAL", str(DEFAULT_DETECT_INTERVAL)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_DETECT_INTERVAL


def track_hold_frames() -> int:
    raw = os.environ.get("FACE_BLUR_TRACK_HOLD_FRAMES", str(DEFAULT_TRACK_HOLD_FRAMES)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TRACK_HOLD_FRAMES


def low_confidence_threshold() -> float:
    raw = os.environ.get("FACE_BLUR_LOW_CONFIDENCE", str(DEFAULT_LOW_CONFIDENCE)).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LOW_CONFIDENCE


def ensemble_enabled() -> bool:
    raw = os.environ.get("FACE_BLUR_ENSEMBLE", "1" if DEFAULT_ENSEMBLE else "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def ensemble_opencv_ssd_enabled() -> bool:
    """OpenCV SSD adds noisy large boxes on crowds; off by default."""
    raw = os.environ.get("FACE_BLUR_ENSEMBLE_SSD", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def merge_detection_min_score() -> float:
    raw = os.environ.get("FACE_BLUR_MERGE_MIN_SCORE", "0.32").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.32


def redaction_pad_ratio() -> float:
    raw = os.environ.get("FACE_BLUR_REDACTION_PAD", "0.05").strip()
    try:
        return max(0.0, min(float(raw), 0.2))
    except ValueError:
        return 0.05


def head_expand_enabled() -> bool:
    raw = os.environ.get("FACE_BLUR_HEAD_EXPAND", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def byte_track_enabled() -> bool:
    raw = os.environ.get("FACE_BLUR_BYTE_TRACK", "1" if DEFAULT_BYTE_TRACK else "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def redaction_mask() -> str:
    return os.environ.get("FACE_BLUR_REDACTION_MASK", DEFAULT_REDACTION_MASK).strip().lower()


def models_dir() -> Path:
    raw = os.environ.get("FACE_BLUR_MODELS_DIR", "").strip()
    if raw:
        return Path(raw)
    media = os.environ.get("MEDIA_ROOT", "").strip()
    if media:
        return Path(media) / "face_blur_models"
    return Path(tempfile.gettempdir()) / "face_blur_models"


def onnx_model_path() -> str | None:
    p = os.environ.get("FACE_BLUR_ONNX_MODEL_PATH", "").strip()
    return p or None


def manifest_path() -> str | None:
    p = os.environ.get("FACE_BLUR_MODEL_MANIFEST_PATH", "").strip()
    return p or None


def model_version() -> str | None:
    v = os.environ.get("FACE_BLUR_MODEL_VERSION", "").strip()
    return v or None


# --- License plate redaction (local ONNX YOLO) ---

DEFAULT_PLATE_CONFIDENCE = 0.35
DEFAULT_PLATE_IOU = 0.45
DEFAULT_PLATE_INPUT_SIZE = 640


def plate_onnx_model_path() -> str | None:
    p = os.environ.get("PRIVACY_PLATE_ONNX_MODEL_PATH", "").strip()
    return p or None


def plate_onnx_model_url() -> str | None:
    p = os.environ.get("PRIVACY_PLATE_ONNX_MODEL_URL", "").strip()
    return p or None


def plate_auto_download_enabled() -> bool:
    raw = os.environ.get("PRIVACY_PLATE_AUTO_DOWNLOAD", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def plate_redaction_enabled() -> bool:
    """True when an explicit plate model path is set or auto-download is allowed."""
    if plate_onnx_model_path():
        return True
    from .lpd_yunet import bundled_model_path

    return bundled_model_path().is_file() or plate_auto_download_enabled()


def plate_confidence_threshold() -> float:
    raw = os.environ.get("PRIVACY_PLATE_CONFIDENCE", str(DEFAULT_PLATE_CONFIDENCE)).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_PLATE_CONFIDENCE


def plate_iou_threshold() -> float:
    raw = os.environ.get("PRIVACY_PLATE_IOU", str(DEFAULT_PLATE_IOU)).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_PLATE_IOU


def plate_input_size() -> int:
    raw = os.environ.get("PRIVACY_PLATE_INPUT_SIZE", str(DEFAULT_PLATE_INPUT_SIZE)).strip()
    try:
        return max(320, int(raw))
    except ValueError:
        return DEFAULT_PLATE_INPUT_SIZE


def plate_class_ids() -> tuple[int, ...] | None:
    """Optional YOLO class filter; None keeps all classes (single-class plate models)."""
    raw = os.environ.get("PRIVACY_PLATE_CLASS_IDS", "").strip()
    if not raw:
        return None
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return tuple(ids) if ids else None


def local_plate_redaction_available() -> bool:
    """Whether the local stack can redact plates without Sightengine."""
    if plate_onnx_model_path():
        from pathlib import Path

        return Path(plate_onnx_model_path()).is_file()
    from .lpd_yunet import bundled_model_path

    if bundled_model_path().is_file():
        return True
    from .plate_model_assets import plate_model_path

    if plate_model_path().is_file():
        return True
    return bool(plate_auto_download_enabled() and plate_onnx_model_url()
                and os.environ.get("PRIVACY_PLATE_ONNX_SHA256", "").strip())
