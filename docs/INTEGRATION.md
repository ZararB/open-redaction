# Integrating Open Redaction

Open Redaction is a normal Python package. You can use it as a CLI, import the engine in your app, or run it in a container/worker.

## Install

From Git (private or public):

```bash
pip install "git+https://github.com/ZararB/open-redaction.git"
```

For local development:

```bash
git clone https://github.com/ZararB/open-redaction.git
cd open-redaction
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements: Python 3.10+, **ffmpeg** / **ffprobe** on `PATH`.

## CLI

```bash
open-redact input.mp4 -o output.redacted.mp4 --json
open-redact input.mp4 -o out.mp4 --no-plates
open-redact input.mp4 -o out.mp4 --no-faces
```

## Python API (recommended)

```python
from agents.ml.face_blur.unified_privacy_redaction import UnifiedPrivacyRedactionService

service = UnifiedPrivacyRedactionService.from_env(
    blur_faces=True,
    blur_plates=True,
)
result = service.redact_video_fast("/path/in.mp4", "/path/out.mp4")
print(result)  # targets, model digests, timing
```

Thin facade (same behavior):

```python
from agents.services.privacy_redaction_service import PrivacyRedactionService

PrivacyRedactionService(blur_faces=True, blur_plates=True).redact_video("in.mp4", "out.mp4")
```

## Environment tuning

Copy [.env.example](../.env.example). Common variables:

| Variable | Purpose |
|----------|---------|
| `FACE_BLUR_DETECTOR` | `yunet` (default offline), `scrfd`, `auto`, `max` |
| `FACE_BLUR_STYLE` | `blackout` or `gaussian` |
| `FACE_BLUR_BYTE_TRACK` | `1` for temporal tracking |
| `PRIVACY_PLATE_ONNX_MODEL_PATH` | Optional custom plate YOLO ONNX |
| `FACE_BLUR_MODELS_DIR` | Cache dir for downloaded optional weights |

See `agents/ml/face_blur/config.py` for the full list.

## Models and hosting

| Model | Size | Default delivery |
|-------|------|------------------|
| YuNet face | ~230 KB | **In git** (`agents/ml/face_blur/models/`) |
| LPD-YuNet plate | ~4 MB | **In git** (same directory) |
| SCRFD-2.5G | ~10 MB | **Download on first use** from Hugging Face ([RuteNL/SCRFD-face-detection-ONNX](https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX)) |
| OpenCV Res10 SSD | ~10 MB | Downloaded when ensemble SSD is enabled |
| Custom plate YOLO | varies | Your path or HTTPS URL + pinned SHA-256 |

Bundled ONNX files are SHA-256 verified at load time. Optional detectors need network unless you pre-populate `FACE_BLUR_MODELS_DIR`.

For air-gapped deploys, vendor the SCRFD file into `FACE_BLUR_MODELS_DIR/scrfd/scrfd_2.5g_bnkps.onnx` before runtime.

## Docker / workers

Use `agents/ml/face_blur/runpod/bootstrap_models.py` at process startup to verify bundled weights offline (no cloud SDK). Mount `FACE_BLUR_MODELS_DIR` if you cache SCRFD or custom plate models.

## Evaluation before release

```bash
python -m agents.ml.face_blur.evaluation labels.jsonl --dataset-id prod-v1 --output report.json
```

Non-zero exit if precision/recall thresholds fail. See [DATASET_PRIVACY_REDACTION.md](./DATASET_PRIVACY_REDACTION.md).

## Maintainers: sync from a private monorepo

```bash
export OPEN_REDACTION_UPSTREAM=/path/to/your-monorepo/opensource/napknstack-privacy-redaction/agents
python scripts/sync_from_upstream.py
pytest -q
```

Only open-source-safe modules are copied (no Sightengine, RunPod handlers, or Django admin).
