#!/usr/bin/env python3
"""Package a labeled batch for PR / coordinator merge (Leela Zero–style weight/game upload analogue)."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--station-id", required=True, help="Your contributor id (github handle, lab name)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--tarball", type=Path, default=None)
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    labels = batch_dir / "labels.jsonl"
    if not labels.is_file():
        print("labels.jsonl missing — label in studio first", flush=True)
        return 1

    contribution = {
        "schema_version": 1,
        "station_id": args.station_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_dir.name,
        "labels_sha256": sha256_file(labels),
        "notes": args.notes,
    }
    out_json = batch_dir / "contribution.json"
    out_json.write_text(json.dumps(contribution, indent=2) + "\n", encoding="utf-8")

    tar_path = args.tarball or (batch_dir / f"{batch_dir.name}_contribution.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in ("manifest.json", "labels.jsonl", "contribution.json"):
            p = batch_dir / name
            if p.is_file():
                tar.add(p, arcname=f"{batch_dir.name}/{name}")
        frames = batch_dir / "frames"
        if frames.is_dir():
            for png in sorted(frames.glob("*.png")):
                tar.add(png, arcname=f"{batch_dir.name}/frames/{png.name}")

    print(json.dumps({"contribution": str(out_json), "tarball": str(tar_path)}, indent=2))
    print("Open a PR adding labels.jsonl + contribution.json (not raw video).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
