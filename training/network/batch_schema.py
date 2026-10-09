"""Contributor batch manifest helpers (framework-agnostic)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BatchManifest:
    batch_id: str
    source_id: str
    title: str
    source_url: str
    license_note: str
    tags: list[str] = field(default_factory=list)
    frame_count: int = 0
    labeling_status: str = "open"  # open | in_review | sealed
    created_at: str = field(default_factory=utc_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "source": {
                "id": self.source_id,
                "title": self.title,
                "url": self.source_url,
                "license": self.license_note,
                "tags": self.tags,
            },
            "frames": {"count": self.frame_count},
            "labeling": {"status": self.labeling_status, "min_reviewers": 1},
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchManifest:
        src = data.get("source") or {}
        frames = data.get("frames") or {}
        labeling = data.get("labeling") or {}
        return cls(
            batch_id=str(data.get("batch_id") or ""),
            source_id=str(src.get("id") or ""),
            title=str(src.get("title") or ""),
            source_url=str(src.get("url") or ""),
            license_note=str(src.get("license") or ""),
            tags=list(src.get("tags") or []),
            frame_count=int(frames.get("count") or 0),
            labeling_status=str(labeling.get("status") or "open"),
            created_at=str(data.get("created_at") or utc_now()),
            schema_version=int(data.get("schema_version") or 1),
        )


def load_manifest(batch_dir: Path) -> BatchManifest:
    path = batch_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing_manifest:{path}")
    return BatchManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(batch_dir: Path, manifest: BatchManifest) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def labels_path(batch_dir: Path) -> Path:
    return batch_dir / "labels.jsonl"


def pseudo_labels_path(batch_dir: Path) -> Path:
    return batch_dir / "labels_pseudo.jsonl"


def load_labels_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Map relative image path -> row."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("image") or "").replace("\\", "/")
        if key:
            rows[key] = row
    return rows


def write_labels_jsonl(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(rows[k], ensure_ascii=False) for k in sorted(rows.keys())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
