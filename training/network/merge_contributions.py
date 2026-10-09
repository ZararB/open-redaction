#!/usr/bin/env python3
"""Merge human labels from contributor batches into a single training JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.network.batch_schema import load_labels_jsonl, write_labels_jsonl


def merge_batch_dirs(batch_dirs: list[Path], *, prefer: str = "human") -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for batch_dir in batch_dirs:
        batch_id = batch_dir.name
        human = load_labels_jsonl(batch_dir / "labels.jsonl")
        pseudo = load_labels_jsonl(batch_dir / "labels_pseudo.jsonl")
        for key, row in pseudo.items():
            if key not in human:
                row = dict(row)
                row.setdefault("note", "pseudo_only")
                row["batch_id"] = batch_id
                merged[f"{batch_id}/{key}"] = row
        for key, row in human.items():
            row = dict(row)
            row["note"] = row.get("note") or "human_reviewed"
            row["batch_id"] = batch_id
            merged[f"{batch_id}/{key}"] = row
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge contributor batch labels")
    parser.add_argument(
        "contributions_root",
        type=Path,
        nargs="?",
        default=Path("datasets/contributions"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("datasets/contributions/merged/labels.jsonl"),
    )
    args = parser.parse_args()
    root = args.contributions_root
    batches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    if not batches:
        print(f"no batches under {root}")
        return 1
    merged = merge_batch_dirs(batches)
    out_rows: dict[str, dict] = {}
    for composite_key, row in merged.items():
        batch_id = composite_key.split("/", 1)[0]
        new_row = dict(row)
        img = str(new_row.get("image") or "frames/unknown.png")
        if not img.startswith("frames/"):
            img = f"frames/{Path(img).name}"
        new_row["image"] = f"{batch_id}/{img}"
        new_row["contributor_batch"] = batch_id
        out_rows[new_row["image"]] = new_row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_labels_jsonl(args.output, out_rows)
    summary = {
        "batches": len(batches),
        "rows": len(out_rows),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
