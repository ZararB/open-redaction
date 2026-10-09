# NYC Times Square (45s clip) — pseudo-label slice

30 frames extracted every **1.5s** from a consented-style **public YouTube walking clip** (see `samples/redaction_test_samples.json` → `nyc_times_square_night_walk`).

| File | Description |
|------|-------------|
| `frames/` | PNG exports |
| `labels.jsonl` | Empty annotation stubs from `build_eval_frames.py` |
| `labels_pseudo.jsonl` | **Auto labels** from `FACE_BLUR_DETECTOR=max` — **must be human-reviewed** before release registration |

This set is **too small** for `release_registry` (needs ≥100 frames). Use it to tune thresholds and mask coverage on dense night crowds.

Regenerate pseudo labels:

```bash
cd backend && PYTHONPATH=. FACE_BLUR_DETECTOR=max \
  python agents/ml/face_blur/scripts/build_pseudo_labels.py \
  --frames-dir datasets/privacy_redaction/nyc_times_square_45s/frames \
  --labels-out datasets/privacy_redaction/nyc_times_square_45s/labels_pseudo.jsonl \
  --rel-root datasets/privacy_redaction/nyc_times_square_45s
```

Do not treat pseudo labels as ground truth for compliance.
