# Annotation schema (`labels.jsonl`)

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `image` | string | Path to PNG/JPEG, relative to the JSONL directory unless absolute |
| `faces` | `[[x,y,w,h], ...]` | Pixel axis-aligned boxes; empty list = no faces in frame |
| `license_plates` | `[[x,y,w,h], ...]` | Same convention for plates |

Coordinates match OpenCV: origin top-left, `w`/`h` strictly positive for positives.

## Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `note` | string | Provenance (e.g. `unlabeled` exports) |
| `source_video` | string | Clip id or URL for traceability |
| `timestamp_sec` | number | Time in source video |

## Release gate minimums (`release_registry.validate_report`)

- ≥ **100** labeled frames in the file
- Per target: ≥ **50** positive boxes (TP+FN), ≥ **10** negative frames (empty list)
- ≥ **95%** recall and ≥ **90%** precision (configurable in CLI)
- Encoded redaction must fully cover annotated regions (`mask_coverage.passed`)

## Dataset versioning

Set a stable `--dataset-id` when running evaluation (e.g. `open-redaction-eval-v1`). Store the report JSON alongside the labels digest (`annotation_sha256` in the report).
