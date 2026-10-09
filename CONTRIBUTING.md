# Contributing

1. Fork and branch from `main`.
2. `pip install -e ".[dev]"` and run `pytest -q`.
3. Run `./scripts/release_gate.sh` before opening a PR.
4. Do not add proprietary integrations, secrets, or customer-specific URLs.
5. Update [MODEL_PROVENANCE.md](./MODEL_PROVENANCE.md) if bundled ONNX digests change.

Engine changes should be synced from the canonical private tree via `scripts/sync_from_upstream.py` when applicable, then reviewed for accidental leakage (imports of cloud SDKs, hard-coded keys, internal hostnames).

## Training network

Distributed **labeling stations** and optional **train stations** follow [training/TRAINING_NETWORK.md](./training/TRAINING_NETWORK.md). Submit human-reviewed `labels.jsonl` via PR; do not commit raw YouTube downloads (`.cache/` is gitignored).
