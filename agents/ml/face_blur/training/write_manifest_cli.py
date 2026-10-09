#!/usr/bin/env python3
"""
Write a versioned manifest JSON for a face-blur ONNX artifact (sha256 + metadata).

Example (from repo backend/ with PYTHONPATH including agents parent):

  PYTHONPATH=backend python agents/ml/face_blur/training/write_manifest_cli.py \\
    --onnx /models/face_ssd.onnx \\
    --out /models/face_ssd.manifest.json \\
    --version 2026.05.12-1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write face-blur ONNX manifest")
    parser.add_argument("--onnx", required=True, help="Path to .onnx model")
    parser.add_argument("--out", required=True, help="Output manifest JSON path")
    parser.add_argument("--version", required=True, help="Model version label")
    parser.add_argument("--input-size", type=int, default=300)
    args = parser.parse_args()

    from agents.ml.face_blur.manifest import write_manifest

    write_manifest(
        args.onnx,
        args.out,
        model_version=args.version,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
