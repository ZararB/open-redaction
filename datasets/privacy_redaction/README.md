# Privacy redaction evaluation datasets

Hold-out **frame annotations** for the release gate in `agents/ml/face_blur/evaluation.py`.

## Layout

| Path | Purpose |
|------|---------|
| `toy_ci/` | Synthetic PNGs + `labels.jsonl` for CI (no real biometric data) |
| `extracted/` | Output of `build_eval_frames.py` (unlabeled exports — annotate before eval) |
| `schema.md` | JSONL field definitions |

## JSONL format

One object per line:

```json
{"image":"frames/0001.png","faces":[[x,y,w,h]],"license_plates":[[x,y,w,h]]}
```

Both `faces` and `license_plates` are required; use `[]` for negatives.

## Commands

```bash
# Regenerate synthetic CI corpus (120 frames)
python backend/agents/ml/face_blur/scripts/generate_toy_eval_dataset.py

# Export frames from a bundled test sample (unlabeled stub JSONL)
python backend/agents/ml/face_blur/scripts/build_eval_frames.py \
  --sample-id oslo_tram_street --max-frames 20

# Run evaluation gate
python -m agents.ml.face_blur.evaluation \
  backend/datasets/privacy_redaction/toy_ci/labels.jsonl \
  --dataset-id toy-ci-v1 --output /tmp/report.json
```

Do **not** commit customer video or identifiable imagery without consent and a dataset card.
