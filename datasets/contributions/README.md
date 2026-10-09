# Contributor batches

Labeled frames from the **Open Redaction training network** live here—one directory per public clip batch.

| Path | Purpose |
|------|---------|
| `batch_<slug>/frames/` | PNG exports (git: optional; regenerate via download script) |
| `batch_<slug>/labels.jsonl` | **Human** ground truth — commit when reviewed |
| `batch_<slug>/labels_pseudo.jsonl` | Machine bootstrap — do not treat as GT |
| `batch_<slug>/contribution.json` | Station id + SHA-256 for coordinator merge |
| `.cache/` | Downloaded WebM/MP4 — **gitignored** |

Bootstrap:

```bash
./training/scripts/station_setup.sh
```

Hand-label the first batches in the studio, then submit via `training/scripts/submit_contribution.py`.
