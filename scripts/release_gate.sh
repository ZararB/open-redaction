#!/usr/bin/env bash
# Standalone release gate for open-redaction (no monorepo / Django required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
VENV="${TMPDIR:-/tmp}/open-redaction-gate-$$"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -e "$ROOT[dev]"
cd "$ROOT"
pytest -q tests/test_evaluation_dataset.py tests/test_privacy_redaction.py
WHEEL_DIR="${TMPDIR:-/tmp}/open-redaction-wheels-$$"
mkdir -p "$WHEEL_DIR"
pip wheel "$ROOT" --no-deps -w "$WHEEL_DIR"
CHECK_VENV="${TMPDIR:-/tmp}/open-redaction-wheel-$$"
"$PY" -m venv "$CHECK_VENV"
"$CHECK_VENV/bin/pip" install -q "$WHEEL_DIR"/open_redaction-*.whl
( cd /tmp && "$CHECK_VENV/bin/open-redact" --help >/dev/null )
echo "release_gate_ok wheel=$WHEEL_DIR"
