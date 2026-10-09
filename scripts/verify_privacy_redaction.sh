#!/usr/bin/env bash
# Large-scale privacy redaction verification for open-redaction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY:-python3}"
fi

export PYTHONPATH="$ROOT"
export FACE_BLUR_DETECTOR="${FACE_BLUR_DETECTOR:-max}"
STRICT_EVAL="${STRICT_EVAL:-0}"
VERIFY_SKIP_FRAME_BUILD="${VERIFY_SKIP_FRAME_BUILD:-0}"

BENCH_DIR="$ROOT/datasets/privacy_redaction/nyc_times_square_benchmark"
REPORT_DIR="$ROOT/datasets/privacy_redaction/reports"
NYC_SOURCE="${NYC_SOURCE:-}"

mkdir -p "$REPORT_DIR"

echo "==> Unit + integration tests (agents, no Django)"
"$PY" -m pytest -o addopts='' -p no:django --noconftest -q \
  "$ROOT/tests/test_evaluation_dataset.py" \
  "$ROOT/tests/test_privacy_redaction.py" \
  "$ROOT/tests/test_bootstrap_models.py" \
  "$ROOT/tests/test_mask_coverage.py" \
  "$ROOT/tests/test_byte_tracker.py" \
  "$ROOT/tests/test_face_blur_boxes.py" \
  "$ROOT/tests/test_manual_masks.py" \
  "$ROOT/tests/test_face_blur_tracking.py"

echo "==> Toy CI evaluation gate (synthetic 120 frames, mirror detectors)"
"$PY" -m pytest -o addopts='' -p no:django --noconftest -q \
  "$ROOT/tests/test_evaluation_dataset.py::test_evaluation_passes_on_toy_with_mirror_detectors"

if [[ -f "$NYC_SOURCE" ]]; then
  if [[ "$VERIFY_SKIP_FRAME_BUILD" == "1" ]] && [[ -f "$BENCH_DIR/frames/nyc_0119.png" ]]; then
    echo "==> SKIP frame build (VERIFY_SKIP_FRAME_BUILD=1, 120 frames present)"
  else
  echo "==> Building NYC benchmark frames (120 @ 1s from t=120s)"
  "$PY" "$ROOT/agents/ml/face_blur/scripts/build_eval_frames.py" \
    --video "$NYC_SOURCE" \
    --out "$BENCH_DIR" \
    --start-sec 120 \
    --every-sec 1.0 \
    --max-frames 120 \
    --prefix nyc
  fi

  echo "==> Pseudo-labeling benchmark (review required for ground truth)"
  "$PY" "$ROOT/agents/ml/face_blur/scripts/build_pseudo_labels.py" \
    --frames-dir "$BENCH_DIR/frames" \
    --labels-out "$BENCH_DIR/labels_pseudo.jsonl" \
    --rel-root "$BENCH_DIR"

  echo "==> Real-model evaluation + redaction smoke"
  REPORT="$REPORT_DIR/nyc_times_square_benchmark_$(date +%Y%m%d).json"
  set +e
  "$PY" -m agents.ml.face_blur.evaluation \
    "$BENCH_DIR/labels_pseudo.jsonl" \
    --dataset-id nyc-times-square-benchmark-v1 \
    --min-recall 0.85 \
    --min-precision 0.85 \
    --output "$REPORT"
  EVAL_RC=$?
  set -e
  echo "evaluation_exit_code=$EVAL_RC report=$REPORT"
  if [[ "$STRICT_EVAL" == "1" ]] && [[ "$EVAL_RC" -ne 0 ]]; then
    echo "STRICT_EVAL=1: NYC benchmark evaluation failed (see report)"
    exit "$EVAL_RC"
  fi
  "$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())
print("frames", report.get("frames"))
for target, m in (report.get("metrics") or {}).items():
    print(f"  {target}: recall={m.get('recall'):.3f} precision={m.get('precision'):.3f} neg={m.get('negative_frames')}")
mask = report.get("mask_coverage") or {}
print("  mask_coverage.passed", mask.get("passed"), "misses", mask.get("annotation_misses"))
print("  overall.passed", report.get("passed"))
PY

  SMOKE_IN="$BENCH_DIR/frames/nyc_0000.png"
  if [[ -f "$SMOKE_IN" ]]; then
    SMOKE_MP4="$REPORT_DIR/smoke_one_frame.mp4"
    SMOKE_OUT="$REPORT_DIR/smoke_one_frame_redacted.mp4"
    ffmpeg -y -hide_banner -loglevel error -loop 1 -i "$SMOKE_IN" -t 1 -pix_fmt yuv420p "$SMOKE_MP4"
    "$PY" -m agents.ml.face_blur.privacy_cli "$SMOKE_MP4" -o "$SMOKE_OUT" --json
    echo "smoke_redacted=$SMOKE_OUT"
  fi
else
  echo "SKIP NYC benchmark (set NYC_SOURCE or place source_full.mp4 at default path)"
fi

echo "verify_privacy_redaction_ok"
