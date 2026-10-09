#!/usr/bin/env bash
# Optional: train a small YOLO detector on WIDERFace and export ONNX.
# The default inference path (OnnxSsdFaceDetector) expects SSD-style tensors [...,7]
# (same as OpenCV Res10_300x300). YOLO ONNX requires a separate decoder — use this job
# to produce weights, then add a YOLO-specific detector or export SSD-compatible ONNX.
set -euo pipefail
if ! command -v yolo >/dev/null 2>&1; then
  echo "install ultralytics: pip install ultralytics" >&2
  exit 1
fi
DATA_YAML="${DATA_YAML:-/data/widerface/widerface.yaml}"
yolo task=detect mode=train model=yolov8n.pt data="${DATA_YAML}" epochs=50 imgsz=640 project=/out name=face_blur
yolo task=detect mode=export model="/out/face_blur/weights/best.pt" format=onnx imgsz=640
echo "ONNX under /out/face_blur/weights/ — copy to blob storage and set FACE_BLUR_ONNX_MODEL_PATH"
