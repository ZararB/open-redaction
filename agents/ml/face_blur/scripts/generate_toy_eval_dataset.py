"""Generate a synthetic JSONL + PNG dataset for privacy-redaction CI (no real faces)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

SEED = 42


def _draw_face_blob(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    cv2.ellipse(frame, (cx, cy), (w // 2, h // 2), 0, 0, 360, (60, 80, 220), -1)
    cv2.ellipse(frame, (cx - w // 6, cy - h // 8), (w // 10, h // 12), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(frame, (cx + w // 6, cy - h // 8), (w // 10, h // 12), 0, 0, 360, (20, 20, 20), -1)


def _draw_plate_blob(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 240, 240), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), 2)
    cv2.putText(
        frame,
        "ABC123",
        (x + 4, y + h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )


def generate(out_dir: Path, *, frames: int, width: int, height: int) -> Path:
    if frames < 100:
        raise ValueError("toy_dataset_requires_at_least_100_frames_for_registry_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    rng = random.Random(SEED)

    face_pos_target = max(55, int(frames * 0.5))
    plate_pos_target = max(55, int(frames * 0.5))
    neg_target = max(12, int(frames * 0.1))

    rows: list[dict] = []
    for idx in range(frames):
        img = np.full((height, width, 3), 180, dtype=np.uint8)
        faces: list[list[int]] = []
        plates: list[list[int]] = []

        face_positive = idx < face_pos_target
        plate_positive = idx < plate_pos_target
        if face_pos_target <= idx < face_pos_target + neg_target:
            face_positive = False
        if plate_pos_target <= idx < plate_pos_target + neg_target:
            plate_positive = False

        if face_positive:
            fx = rng.randint(20, width - 80)
            fy = rng.randint(20, height - 70)
            fw = rng.randint(36, 56)
            fh = rng.randint(44, 64)
            faces = [[fx, fy, fw, fh]]
            _draw_face_blob(img, (fx, fy, fw, fh))

        if plate_positive:
            px = rng.randint(20, width - 120)
            py = rng.randint(height // 2, height - 40)
            pw = rng.randint(70, 110)
            ph = rng.randint(18, 28)
            plates = [[px, py, pw, ph]]
            _draw_plate_blob(img, (px, py, pw, ph))

        name = f"frame_{idx:04d}.png"
        cv2.imwrite(str(frames_dir / name), img)
        rows.append({"image": f"frames/{name}", "faces": faces, "license_plates": plates})

    labels_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return labels_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic privacy-redaction eval dataset")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "datasets" / "privacy_redaction" / "toy_ci",  # backend root
        help="Output directory (contains frames/ + labels.jsonl)",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args(argv)
    path = generate(args.out, frames=args.frames, width=args.width, height=args.height)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
