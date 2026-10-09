"""Write labels.jsonl from detector predictions on exported frames (human review required)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

_BACKEND_ROOT = Path(__file__).resolve().parents[4]


def _repo_root() -> Path:
    if (_BACKEND_ROOT / "datasets").is_dir():
        return _BACKEND_ROOT
    return Path(__file__).resolve().parents[5]


def pseudo_label(frames_dir: Path, labels_out: Path, *, rel_root: Path) -> None:
    os.environ.setdefault("FACE_BLUR_DETECTOR", "max")
    from agents.ml.face_blur.factory import build_detector
    from agents.ml.face_blur.plate_detector import build_plate_detector_if_configured

    face_det = build_detector("onnx", 0.25)
    plate_det = build_plate_detector_if_configured()
    if plate_det is None:
        raise RuntimeError("license_plate_detector_unavailable")

    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no_png_frames:{frames_dir}")

    lines: list[str] = []
    for png in pngs:
        frame = cv2.imread(str(png))
        if frame is None:
            raise SystemExit(f"unreadable:{png}")
        faces = []
        for box, _score in face_det.detect_scored(frame):
            x, y, w, h = (int(v) for v in box)
            if w > 0 and h > 0:
                faces.append([x, y, w, h])
        plates = []
        for box, _score in plate_det.detect_scored(frame):
            x, y, w, h = (int(v) for v in box)
            if w > 0 and h > 0:
                plates.append([x, y, w, h])
        rel = png.relative_to(rel_root)
        lines.append(
            json.dumps(
                {
                    "image": str(rel).replace("\\", "/"),
                    "faces": faces,
                    "license_plates": plates,
                    "note": "pseudo_label — review before release gate",
                }
            )
        )

    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(lines), "labels": str(labels_out)}, indent=2))


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Pseudo-label exported PNG frames")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=root / "datasets" / "privacy_redaction" / "extracted" / "frames",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=root / "datasets" / "privacy_redaction" / "extracted" / "labels_pseudo.jsonl",
    )
    parser.add_argument(
        "--rel-root",
        type=Path,
        default=root / "datasets" / "privacy_redaction" / "extracted",
    )
    args = parser.parse_args(argv)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    pseudo_label(args.frames_dir, args.labels_out, rel_root=args.rel_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
