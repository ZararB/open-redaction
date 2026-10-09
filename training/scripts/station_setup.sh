#!/usr/bin/env bash
# Bootstrap a self-hosted Open Redaction labeling + training station.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Open Redaction contributor station setup"
if ! command -v ffmpeg >/dev/null; then
  echo "Install ffmpeg first (brew install ffmpeg / apt install ffmpeg)" >&2
  exit 1
fi

PY="${PY:-python3}"
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q -e ".[dev,youtube]"

STATION_ID="${STATION_ID:-$(whoami)-$(hostname -s 2>/dev/null || echo local)}"
mkdir -p datasets/contributions
echo "$STATION_ID" > datasets/contributions/.station_id
echo "station_id=$STATION_ID"

echo "==> Download Commons batches (no YouTube unless INCLUDE_YOUTUBE=1)"
if [[ "${INCLUDE_YOUTUBE:-0}" == "1" ]]; then
  python training/scripts/download_training_batches.py --include-youtube --sample-id nyc_times_square_night_walk
fi
python training/scripts/download_training_batches.py --all-commons

echo "==> Optional pseudo-label first batch (requires onnxruntime + models)"
FIRST_BATCH="$(find datasets/contributions -maxdepth 1 -type d -name 'batch_*' | head -1 || true)"
if [[ -n "$FIRST_BATCH" ]]; then
  python training/scripts/prelabel_batch.py "$FIRST_BATCH" || echo "prelabel skipped (install models / onnxruntime)"
fi

cat <<EOF

Station ready.

  Label:  open-redact-studio --batch $FIRST_BATCH
  Verify: python -m agents.ml.face_blur.evaluation \\
            $FIRST_BATCH/labels.jsonl --dataset-id local-v1 -o /tmp/eval.json

  Submit: python training/scripts/submit_contribution.py $FIRST_BATCH --station-id $STATION_ID

See training/TRAINING_NETWORK.md
EOF
