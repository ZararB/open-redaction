# Open Redaction training network (Leela Zero–style)

Open Redaction improves when **many stations** contribute **labeled frames** and **exported ONNX weights**, coordinated through a **public golden dataset** and **evaluation gate**—similar to how Leela Zero distributed self-play games and weight uploads, but for privacy detection.

## Roles

| Role | What you run | Output |
|------|----------------|--------|
| **Labeling station** | `station_setup.sh` + `open-redact-studio` | Human-reviewed `labels.jsonl` per batch |
| **Prelabel worker** | `prelabel_batch.py` | `labels_pseudo.jsonl` (bootstrap only) |
| **Train station** | `train_export_yolo.sh` or your own PyTorch job | ONNX + eval report |
| **Coordinator** (maintainers) | Merge PRs, run `merge_contributions.py`, publish model manifest | New default weights when gate passes |

There is no central GPU cluster requirement: stations are **self-hosted** (laptop, homelab, cloud VM).

## Workflow

1. **Download a batch** (or create one from your own consented video):

   ```bash
   python training/scripts/download_training_batches.py --all-commons
   ```

2. **Optional pseudo-label** (speeds up your hand pass):

   ```bash
   python training/scripts/prelabel_batch.py datasets/contributions/batch_001_oslo_tram
   ```

3. **Label in the studio** (human-in-the-loop):

   ```bash
   open-redact-studio --batch datasets/contributions/batch_001_oslo_tram
   ```

   Draw boxes: **1** = face, **2** = license plate. **N/P** next/prev frame. **S** save. Keyboard-first rect overlays in the labeling studio.

4. **Validate locally**:

   ```bash
   python -m agents.ml.face_blur.evaluation \
     datasets/contributions/batch_001_oslo_tram/labels.jsonl \
     --dataset-id contrib-oslo-v1 --output /tmp/report.json
   ```

5. **Submit contribution** (PR or tarball):

   ```bash
   python training/scripts/submit_contribution.py datasets/contributions/batch_001_oslo_tram \
     --station-id my-lab --notes "first human pass"
   ```

6. **Train** (optional, when enough human labels exist):

   ```bash
   ./training/scripts/train_export_yolo.sh
   ```

## Batch layout

```
datasets/contributions/batch_NNN_<slug>/
  manifest.json          # source, license, frame params, labeling status
  frames/*.png
  labels_pseudo.jsonl    # machine bootstrap (do not treat as GT)
  labels.jsonl           # human ground truth (you create/edit in studio)
  contribution.json      # filled by submit_contribution.py
```

## Coordinator merge rules

- Human `labels.jsonl` **wins** over pseudo for the same `image` path.
- Contributor id + SHA-256 of labels file recorded in `contribution.json`.
- Merged corpus: `datasets/contributions/merged/labels.jsonl` via `merge_contributions.py`.
- A weight change is accepted only if evaluation on **human-labeled** slices passes `release_registry.validate_report`.

## Ethics & data

- Only use media you may process (Commons, your uploads, consented clips).
- Do **not** commit raw YouTube MP4s to git; keep them under `datasets/contributions/.cache/` (gitignored).
- Faces and plates in exports are for **privacy ML**, not identification.

See [CONTRIBUTING.md](../CONTRIBUTING.md) and [studio/README.md](./studio/README.md).
