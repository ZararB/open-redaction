# Privacy redaction datasets (faces + license plates)

This document describes how to build, annotate, and gate **held-out evaluation data** for the local redaction stack (`agents/ml/face_blur`) and future **Open Redaction** releases.

## Why

Engineering for redaction is ahead of **evidence**: without labeled frames we cannot prove recall on small faces, night footage, or regional plates. The evaluation CLI and `release_registry` module encode minimum quality bars before registering ONNX digests in MLOps.

## Workflow

1. **Extract frames** from public/consented clips (`build_eval_frames.py` or manual ffmpeg).
2. **Annotate** boxes in Label Studio, CVAT, or Roboflow; export to JSONL (see `backend/datasets/privacy_redaction/schema.md`).
3. **Evaluate**:

   ```bash
   cd backend
   export PYTHONPATH=$PWD
   python -m agents.ml.face_blur.evaluation \
     datasets/privacy_redaction/<your-set>/labels.jsonl \
     --dataset-id my-held-out-v1 \
     --output reports/my-held-out-v1.json
   ```

4. **Tune** detectors (`FACE_BLUR_DETECTOR`, plate ONNX, thresholds) until the report passes.
5. **Register** (production monorepo only): `register_privacy_redaction_models` with matching model paths and labels digest.

## Synthetic CI corpus

`backend/datasets/privacy_redaction/toy_ci/` is **computer-generated** ( ellipses + fake plates ) for pipeline tests — not a accuracy benchmark.

Regenerate:

```bash
python backend/agents/ml/face_blur/scripts/generate_toy_eval_dataset.py
```

## Real-world benchmark (recommended v1)

| Slice | Frames (target) | Source idea |
|-------|-----------------|-------------|
| Crowd street | 40 | Wikimedia / consented uploads |
| Transit platform | 30 | Existing admin test samples |
| Night / neon | 30 | NYC Times Square trim |
| Plate US/EU | 40 | Dashcam / crossing clips |
| Negatives | 20 | Empty scenes, backs of heads only |

Pseudo-label with `FACE_BLUR_DETECTOR=max`, then human-fix in the admin mask editor pattern.

## Transfer to Open Redaction

Copy `backend/datasets/privacy_redaction/` (minus any private extracts), both scripts under `agents/ml/face_blur/scripts/`, and this doc when syncing the engine to the desktop `open-redaction` repo.

## Licensing

Public Wikimedia URLs are listed in `face_redaction_test_samples.json`. YouTube samples require `yt-dlp` and respect uploader terms. Never add proprietary customer media to git.
