#!/usr/bin/env python3
"""
CI / release helper: verify ONNX loads and matches optional manifest, then exit 0.

Usage:
  python run_export_cycle.py --onnx /path/model.onnx [--manifest /path/manifest.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--manifest", default="")
    args = p.parse_args()

    from agents.ml.face_blur.manifest import sha256_file, validate_onnx_against_manifest
    from agents.ml.face_blur.model_assets import YUNET_SHA256
    from agents.ml.face_blur.detectors import OnnxSsdFaceDetector, YuNetFaceDetector
    import numpy as np

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"missing onnx: {onnx_path}")
    if args.manifest:
        err = validate_onnx_against_manifest(onnx_path, Path(args.manifest))
        if err:
            raise SystemExit(err)
    digest = sha256_file(onnx_path)
    if digest == YUNET_SHA256:
        det = YuNetFaceDetector(str(onnx_path), confidence_threshold=0.3)
    else:
        det = OnnxSsdFaceDetector(str(onnx_path), confidence_threshold=0.9)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame)
    print("ok", det.name)


if __name__ == "__main__":
    main()
