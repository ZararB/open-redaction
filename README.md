# Open Redaction

**Open Redaction** is a standalone, offline-capable **face and license-plate video redaction** engine for privacy-safe AI workflows. It ships the same local ONNX/OpenCV stack used in production privacy pipelines, without any proprietary cloud routing, API keys, or company infrastructure.

Use it to redact sensitive pixels **on your machine** before training, evaluation, or public release of video data.

## What this repo includes

| Component | Purpose |
|-----------|---------|
| **Unified privacy engine** | Face detection (YuNet + optional SCRFD ensemble), ByteTrack-style tracking, opaque head/body masks |
| **License plates** | Bundled OpenCV Zoo LPD-YuNet (optional custom YOLO ONNX override) |
| **CLI** | `open-redact` — process MP4/WebM locally |
| **Evaluation** | Frame-level precision/recall gate from JSONL annotations |
| **Bundled models** | SHA-256 verified ONNX weights (see [MODEL_PROVENANCE.md](./MODEL_PROVENANCE.md)) |
| **Tests** | Pytest suite for detectors, tracking, masks, bootstrap |
| **Training network** | Leela Zero–style contributor stations, batch downloads, labeling studio |
| **HITL studio** | `open-redact-studio` — bbox labeling on contributor batches |

## Contributor training (distributed)

Anyone can run a **labeling station** on public Commons clips, hand-correct pseudo-labels, and submit `labels.jsonl` for merge—similar to distributed Go/Zero weight contributions, but for privacy detection data.

```bash
./training/scripts/station_setup.sh
open-redact-studio --batch datasets/contributions/batch_oslo_tram_street
```

Full protocol: [training/TRAINING_NETWORK.md](./training/TRAINING_NETWORK.md).

## What is intentionally excluded

To keep this repository safe to publish:

- No Sightengine, RunPod, AWS Secrets Manager, or paid transform integrations
- No Django/Celery admin apps or internal API keys
- No proprietary serverless worker handlers or deployment scripts
- No customer data, manifests tied to private buckets, or internal URLs

You can still deploy the **same local engine** in your own worker; see `agents/ml/face_blur/runpod/bootstrap_models.py` for offline model verification at process startup (generic worker bootstrap only).

---

## Requirements

- **Python 3.10+**
- **ffmpeg** and **ffprobe** on `PATH` (video decode/encode)
- ~200MB disk for venv + bundled ONNX (~5MB models + SCRFD cache if enabled)

### macOS

```bash
brew install ffmpeg python@3.11
```

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3-venv python3-pip
```

---

## Quick start

```bash
git clone https://github.com/ZararB/open-redaction.git
cd open-redaction
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Example usage (input → output)

### 1. Local file (default: faces + plates)

| | Path |
|---|------|
| **Input** | `my_clip.mp4` (any local MP4/WebM/MOV) |
| **Output** | `my_clip.redacted.mp4` (same resolution; audio copied when present) |

```bash
open-redact my_clip.mp4 -o my_clip.redacted.mp4
```

Without `--json`, the CLI prints a one-line summary on stdout and a review reminder on stderr:

```text
Wrote my_clip.redacted.mp4 targets=['face', 'license-plate']
Review the output before upload; detector completion does not prove every face or plate was found.
```

With `--json`, stdout is a machine-readable report (model digests, frame counts, targets):

```bash
open-redact my_clip.mp4 -o my_clip.redacted.mp4 --json
```

```json
{
  "input_path": "my_clip.mp4",
  "output_path": "my_clip.redacted.mp4",
  "total_faces_blurred": 42,
  "total_plates_redacted": 7,
  "total_boxes_redacted": 49,
  "frames_processed": 900,
  "success": true,
  "detector": "YuNetFaceDetector(face_detection_yunet_2023mar.onnx)",
  "plate_detector": "OpenCVZooLPD-YuNet-2023mar",
  "model_artifacts": [
    {
      "target": "face",
      "artifact_sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    },
    {
      "target": "license_plate",
      "artifact_sha256": "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7"
    }
  ],
  "review_required": true,
  "targets": ["face", "license-plate"]
}
```

Counts vary by clip; `review_required` is always `true` — treat redaction as assistive, not a compliance guarantee.

### 2. Public sample clip (Wikimedia)

Catalog: [`samples/redaction_test_samples.json`](samples/redaction_test_samples.json). Example — Oslo tram (faces + street):

```bash
mkdir -p samples/cache
curl -L -o samples/cache/oslo_tram_street.webm \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/60/Tram_in_Oslo_20250107_142906.webm/Tram_in_Oslo_20250107_142906.webm.480p.vp9.webm"

open-redact samples/cache/oslo_tram_street.webm \
  -o samples/cache/oslo_tram_street.redacted.webm \
  --json
```

| Sample ID | Good for | Suggested output |
|-----------|----------|------------------|
| `oslo_tram_street` | Faces, street | `*.redacted.webm` |
| `prague_bus_roundabout` | License plates | `*.redacted.webm` |
| `boston_green_line` | Faces + transit | `*.redacted.webm` |

Or use the helper script with a sample id:

```bash
python agents/ml/face_blur/scripts/build_eval_frames.py \
  --sample-id oslo_tram_street \
  --max-frames 15 \
  --out /tmp/oslo_eval
# → /tmp/oslo_eval/_cache/oslo_tram_street.webm (download) + extracted PNG frames
```

### 3. Target toggles

```bash
open-redact my_clip.mp4 -o faces_only.mp4 --no-plates
open-redact dashcam.mp4 -o plates_only.mp4 --no-faces
```

### 4. Python API (same engine)

```python
from agents.ml.face_blur.unified_privacy_redaction import UnifiedPrivacyRedactionService

result = UnifiedPrivacyRedactionService.from_env().redact_video_fast(
    "my_clip.mp4",
    "my_clip.redacted.mp4",
)
print(result["output_path"], result["targets"])
```

Module equivalent to the CLI:

```bash
python -m agents.ml.face_blur my_clip.mp4 -o my_clip.redacted.mp4 --json
```

**Always review output before publishing.** High recall is not guaranteed on all regions, lighting, or plate styles.

---

## Recommended production-parity settings

For dense street scenes (e.g. crowds), use the full local ensemble:

```bash
export FACE_BLUR_DETECTOR=max
export FACE_BLUR_STYLE=blackout
export FACE_BLUR_REDACTION_MASK=rect
export FACE_BLUR_BYTE_TRACK=1
export FACE_BLUR_ENSEMBLE=1
export FACE_BLUR_ENSEMBLE_SSD=0
export PRIVACY_PLATE_CONFIDENCE=0.35
```

Copy [.env.example](./.env.example) to `.env` and `source` it if you prefer.

---

## Models

### Faces

- **YuNet** (OpenCV Zoo) — bundled, offline default
- **SCRFD-2.5G** — optional; requires `onnxruntime` and may download weights when `FACE_BLUR_DETECTOR=scrfd|auto|max` and network is allowed

### License plates

- **LPD-YuNet** — bundled; trained primarily on Chinese plates per upstream model card — **validate on your geography**

Custom plate detector (your ONNX export):

```bash
export PRIVACY_PLATE_ONNX_MODEL_PATH=/path/to/plate.onnx
export PRIVACY_PLATE_ONNX_SHA256=<64-char hex digest>
```

---

## Evaluation gate

Annotate held-out frames as JSONL (one object per image):

```json
{"image":"frame.png","faces":[[x,y,w,h]],"license_plates":[]}
```

Run:

```bash
python -m agents.ml.face_blur.evaluation labels.jsonl --dataset-id my-v1 --output report.json
```

Non-zero exit if configured precision/recall thresholds fail.

---

## Public test samples

Curated URLs (Wikimedia + optional YouTube) live in [`samples/redaction_test_samples.json`](samples/redaction_test_samples.json). See **Example usage** above for a full download → redact flow.

YouTube entries need optional deps:

```bash
pip install -e ".[youtube]"
python -m yt_dlp -f "bv*[height<=720]+ba/b" -o samples/cache/nyc_walk.mp4 "<youtube-url>"
open-redact samples/cache/nyc_walk.mp4 -o samples/cache/nyc_walk.redacted.mp4 --json
```

---

## Evaluation datasets

See [docs/DATASET_PRIVACY_REDACTION.md](./docs/DATASET_PRIVACY_REDACTION.md) and `datasets/privacy_redaction/`.

```bash
# Synthetic CI corpus (120 frames)
python agents/ml/face_blur/scripts/generate_toy_eval_dataset.py

# Export + pseudo-label real clips (review before trusting metrics)
python agents/ml/face_blur/scripts/build_eval_frames.py --video clip.mp4 --max-frames 30
PYTHONPATH=. FACE_BLUR_DETECTOR=max python agents/ml/face_blur/scripts/build_pseudo_labels.py

python -m agents.ml.face_blur.evaluation datasets/.../labels_pseudo.jsonl \
  --dataset-id my-v1 --output /tmp/report.json
```

Bundled slice: `datasets/privacy_redaction/nyc_times_square_45s/` (30 pseudo-labeled frames from a Times Square walk).

## Development

```bash
pip install -e ".[dev]"
pytest -q
./scripts/release_gate.sh
```

### Sync engine from a private monorepo (maintainers only)

If you maintain a private upstream checkout with the canonical engine:

```bash
export OPEN_REDACTION_UPSTREAM=/path/to/monorepo/backend/agents
python scripts/sync_from_upstream.py
pytest -q
```

This copies only the open-source-safe Python modules and bundled ONNX licenses — never `.env`, secrets, or cloud handlers.

---

## Architecture

See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## Security

See [SECURITY.md](./SECURITY.md). **Do not commit API keys, customer video, or `.env`.**

## Integrate in your project

See [docs/INTEGRATION.md](./docs/INTEGRATION.md) for `pip install`, Python API, model hosting (git vs Hugging Face), and worker bootstrap.

## License

Apache-2.0 — see [LICENSE](./LICENSE).
