# Labeling studio

Browser-based **human-in-the-loop** bbox editor for contributor batches. Canvas overlays, keyboard shortcuts, and a save workflow for **training labels** (`faces` + `license_plates` JSONL), not temporal redaction masks.

## Run

```bash
cd open-redaction
source .venv/bin/activate
open-redact-studio --batch datasets/contributions/batch_oslo_tram_street
```

Open http://127.0.0.1:8765/

## Output

Edits write to `labels.jsonl` in the batch directory (schema: `datasets/privacy_redaction/schema.md`).

Pseudo-labels in `labels_pseudo.jsonl` are never overwritten; the studio reads human file when present.
