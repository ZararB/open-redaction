# Bundled model provenance

| Target | Upstream artifact | SHA-256 | License |
| --- | --- | --- | --- |
| Face | [OpenCV Zoo YuNet 2023mar](https://huggingface.co/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | See `agents/ml/face_blur/models/FACE_YUNET_LICENSE` |
| Plate | [OpenCV Zoo LPD-YuNet 2023mar](https://huggingface.co/opencv/opencv_zoo/tree/main/models/license_plate_detection_yunet) | `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7` | Apache 2.0; see `agents/ml/face_blur/models/LPD_YUNET_LICENSE` |

Both digests are checked when loaded. They are included in CLI results for reproducibility. The plate model's upstream card states it was trained on Chinese plates and may have limited detection on other regions. Passing a local smoke test does not establish privacy recall; the release gate needs held-out annotations covering the intended regions and capture conditions.

## Optional detectors (not vendored in git)

| Model | Source | Notes |
| --- | --- | --- |
| SCRFD-2.5G ONNX | [Hugging Face: RuteNL/SCRFD-face-detection-ONNX](https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX) | Fetched when `FACE_BLUR_DETECTOR` is `scrfd`, `auto`, or `max` and network is allowed |
| OpenCV Res10 SSD | OpenCV GitHub raw URLs | Optional ensemble member |
| Custom plate YOLO | Operator-supplied ONNX | Set `PRIVACY_PLATE_ONNX_MODEL_PATH` or HTTPS download with `PRIVACY_PLATE_ONNX_SHA256` |

**Policy:** ship small, pinned defaults in git (~4.5 MB total); pull larger optional weights from Hugging Face or your own model registry at install/runtime. For reproducible deploys, mirror SCRFD to private storage and set `FACE_BLUR_MODELS_DIR` in CI.
