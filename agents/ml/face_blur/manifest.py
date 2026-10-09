"""Versioned artifact manifest for face-blur ONNX (ops parity with recommendation JSON artifacts)."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_onnx_against_manifest(onnx_path: str | Path, manifest_path: str | Path) -> Optional[str]:
    """
    Returns None if OK, else an error message string.
    Manifest keys: model_sha256 (hex), optional model_version.
    """
    manifest = load_manifest(manifest_path)
    expected = manifest.get("model_sha256")
    if not expected:
        return "manifest missing model_sha256"
    actual = sha256_file(onnx_path)
    if actual.lower() != str(expected).lower():
        return f"sha256 mismatch: manifest={expected} actual={actual}"
    return None


def write_manifest(
    onnx_path: str | Path,
    out_path: str | Path,
    *,
    model_version: str,
    input_size: int = 300,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    onnx_path = Path(onnx_path)
    payload: Dict[str, Any] = {
        "model_kind": "tilestack_face_ssd_onnx",
        "model_version": model_version,
        "onnx_path": str(onnx_path),
        "model_sha256": sha256_file(onnx_path),
        "input_size": input_size,
        "output_layout": "[1,1,N,7] SSD face (opencv sample compatible)",
    }
    if extra:
        payload.update(extra)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote face-blur manifest %s", out_path)
    return payload
