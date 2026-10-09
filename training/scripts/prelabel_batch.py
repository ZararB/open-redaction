#!/usr/bin/env python3
"""Pseudo-label a contributor batch (bootstrap for human studio review)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: labels_pseudo.jsonl in batch")
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve()
    frames = batch_dir / "frames"
    out = args.out or (batch_dir / "labels_pseudo.jsonl")

    os.environ.setdefault("FACE_BLUR_DETECTOR", "max")
    os.environ.setdefault("PYTHONPATH", str(ROOT))

    script = ROOT / "agents" / "ml" / "face_blur" / "scripts" / "build_pseudo_labels.py"
    subprocess = __import__("subprocess")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--frames-dir",
            str(frames),
            "--labels-out",
            str(out),
            "--rel-root",
            str(batch_dir),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT), "FACE_BLUR_DETECTOR": "max"},
    )
    # Seed empty human file with frame list if missing
    human = batch_dir / "labels.jsonl"
    if not human.is_file():
        rows = []
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["note"] = "awaiting_human_review"
            row["faces"] = row.get("faces") or []
            row["license_plates"] = row.get("license_plates") or []
            rows.append(json.dumps(row))
        human.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"seeded {human} from pseudo (edit in studio)")
    print(json.dumps({"labels_pseudo": str(out), "frames": len(list(frames.glob('*.png')))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
