"""Copy the open-source-safe redaction engine from a private upstream checkout."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _default_upstream_candidates() -> list[Path]:
    """Prefer the standalone OSS wheel tree over the slimmer backend checkout."""
    home = Path.home()
    return [
        home / "Desktop" / "napknstack-ios" / "opensource" / "napknstack-privacy-redaction" / "agents",  # maintainer default
    ]


def _upstream() -> Path:
    raw = os.environ.get("OPEN_REDACTION_UPSTREAM", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_dir():
            raise SystemExit(
                f"upstream agents tree not found: {path}\n"
                "Set OPEN_REDACTION_UPSTREAM=/path/to/agents (directory containing ml/face_blur)"
            )
        return path
    for candidate in _default_upstream_candidates():
        if (candidate / "ml" / "face_blur").is_dir():
            return candidate
    raise SystemExit(
        "No upstream agents tree found.\n"
        "Set OPEN_REDACTION_UPSTREAM=/path/to/agents"
    )


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    source = _upstream()
    dest = PACKAGE_ROOT / "agents"
    for relative in ("__init__.py", "ml/__init__.py", "services/privacy_redaction_service.py"):
        _copy_if_exists(source / relative, dest / relative)

    module_dir = dest / "ml" / "face_blur"
    module_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"handler.py", "Dockerfile"}
    for py_file in sorted((source / "ml" / "face_blur").glob("*.py")):
        if py_file.name in skip_names:
            continue
        shutil.copy2(py_file, module_dir / py_file.name)

    runpod_dest = module_dir / "runpod"
    runpod_dest.mkdir(parents=True, exist_ok=True)
    (runpod_dest / "__init__.py").write_text(
        '"""Offline model bootstrap for long-running workers."""\n',
        encoding="utf-8",
    )
    # Keep generic offline bootstrap in-repo; do not overwrite with cloud worker copies.

    models = module_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    for filename in (
        "license_plate_detection_lpd_yunet_2023mar.onnx",
        "LPD_YUNET_LICENSE",
        "face_detection_yunet_2023mar.onnx",
        "FACE_YUNET_LICENSE",
    ):
        _copy_if_exists(source / "ml" / "face_blur" / "models" / filename, models / filename)

    print(f"Synced engine from {source} -> {module_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
