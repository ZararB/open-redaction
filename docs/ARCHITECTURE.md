# Architecture

Open Redaction implements **detect → track → opaque mask → ffmpeg mux**. There is no face recognition, embedding, or identity linkage — only bounding-box privacy redaction.

```
Input video (local path)
    → decode frames (OpenCV / ffmpeg pipeline)
    → face detector(s): YuNet, optional SCRFD ensemble
    → plate detector: LPD-YuNet (per-frame) or optional YOLO ONNX
    → trackers: ByteTrack-style association for temporal stability
    → apply blackout or gaussian masks (configurable)
    → encode silent H.264 MP4 output
```

## Key modules

| Module | Role |
|--------|------|
| `agents/ml/face_blur/unified_privacy_redaction.py` | Faces + plates in one pass |
| `agents/ml/face_blur/factory.py` | Detector construction from env |
| `agents/ml/face_blur/config.py` | Environment tuning |
| `agents/services/privacy_redaction_service.py` | Thin facade for CLI |
| `agents/ml/face_blur/evaluation.py` | Held-out JSONL metrics |
| `agents/ml/face_blur/runpod/bootstrap_models.py` | Optional **offline** worker startup verification (no cloud SDK) |

## Manual masks

Time-varying rectangles/ellipses for QA overrides: `agents/ml/face_blur/manual_masks.py`.

## Training (optional)

`agents/ml/face_blur/training/` contains helpers to export YOLOv8 face models to ONNX — not required for default runtime.

## Failure modes

- Detector errors **propagate** (no silent pass-through of unredacted frames on hard failures in the unified service).
- Plate model may **miss or false-positive** outside its training distribution — human review remains required for compliance use cases.
