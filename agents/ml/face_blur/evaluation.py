"""Annotation-based detector release gate for local privacy models.

JSONL rows: {"image":"/path/frame.png", "faces":[[x,y,w,h]],
"license_plates":[[x,y,w,h]]}. Empty lists are meaningful negatives.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2

from .boxes import box_iou
from .factory import build_detector
from .manifest import sha256_file
from .mask_coverage import evaluate_encoded_frame_masks, small_object_misses, summarize_mask_coverage
from .plate_detector import build_plate_detector_if_configured
from .unified_privacy_redaction import UnifiedPrivacyRedactionService, _model_artifacts


def _detector_predictions(detector, frame, truth_boxes: list) -> list:
    """Run detector; optional test helper `detect_for_annotations(frame, truth)` mirrors GT."""
    fn = getattr(detector, "detect_for_annotations", None)
    if callable(fn):
        return fn(frame, truth_boxes)
    return detector.detect(frame)


def match_boxes(predictions, annotations, threshold: float = 0.5) -> tuple[int, int, int]:
    """One-to-one IoU matching; returns TP, FP, FN."""
    pairs = sorted(
        ((box_iou(pred, truth), pi, ti)
         for pi, pred in enumerate(predictions)
         for ti, truth in enumerate(annotations)),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_annotations: set[int] = set()
    for overlap, pi, ti in pairs:
        if overlap < threshold:
            break
        if pi not in used_predictions and ti not in used_annotations:
            used_predictions.add(pi)
            used_annotations.add(ti)
    tp = len(used_predictions)
    return tp, len(predictions) - tp, len(annotations) - tp


def _source_revision() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def derive_passed(report: dict) -> bool:
    thresholds = report.get("thresholds") or {}
    min_recall = float(thresholds.get("min_recall") or 0)
    min_precision = float(thresholds.get("min_precision") or 0)
    metrics = report.get("metrics") or {}
    detector_ok = all(
        m.get("recall", 0) >= min_recall and m.get("precision", 0) >= min_precision for m in metrics.values()
    )
    mask = report.get("mask_coverage") or {}
    return bool(detector_ok and mask.get("passed") is True)


def evaluate(
    labels_path: Path,
    *,
    min_recall: float,
    min_precision: float,
    face_detector=None,
    plate_detector=None,
) -> dict:
    if not 0 <= min_recall <= 1 or not 0 <= min_precision <= 1:
        raise ValueError("evaluation_threshold_out_of_range")
    face = face_detector or build_detector("opencv", 0.3)
    plate = plate_detector or build_plate_detector_if_configured()
    if plate is None:
        raise RuntimeError("license_plate_detector_unavailable")
    totals = {"face": [0, 0, 0], "license_plate": [0, 0, 0]}
    negative_frames = {"face": 0, "license_plate": 0}
    frames = 0
    mask_rows: list[dict] = []
    redaction_service = UnifiedPrivacyRedactionService.from_env()
    with tempfile.TemporaryDirectory(prefix="privacy-eval-") as tmp:
        tmp_dir = Path(tmp)
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = labels_path.parent / image_path
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"unreadable_annotation_image:{image_path}")
            for key, detector, field in (
                ("face", face, "faces"),
                ("license_plate", plate, "license_plates"),
            ):
                if field not in row:
                    raise ValueError(f"annotation_missing:{field}:{image_path}")
                truth = []
                for box in row[field]:
                    if not isinstance(box, list) or len(box) != 4:
                        raise ValueError(f"invalid_annotation_box:{field}:{image_path}")
                    truth.append(tuple(int(v) for v in box))
                if any(len(box) != 4 or box[2] <= 0 or box[3] <= 0 for box in truth):
                    raise ValueError(f"invalid_annotation_box:{field}:{image_path}")
                if not truth:
                    negative_frames[key] += 1
                counts = match_boxes(_detector_predictions(detector, frame, truth), truth)
                for i, count in enumerate(counts):
                    totals[key][i] += count
                if key == "license_plate":
                    face_truth = [
                        tuple(int(v) for v in box)
                        for box in row.get("faces") or []
                        if isinstance(box, list) and len(box) == 4
                    ]
                    plate_truth = truth
                    encoded = evaluate_encoded_frame_masks(
                        redaction_service,
                        frame,
                        face_boxes=face_truth,
                        plate_boxes=plate_truth,
                        tmp_dir=tmp_dir,
                    )
                    misses = (encoded.get("face_misses") or []) + (encoded.get("plate_misses") or [])
                    encoded["image"] = str(image_path)
                    encoded["small_object_misses"] = small_object_misses(face_truth + plate_truth, misses)
                    encoded["frame_index"] = 0
                    mask_rows.append(encoded)
            frames += 1
    if frames == 0:
        raise ValueError("empty_evaluation_dataset")
    metrics = {}
    for key, (tp, fp, fn) in totals.items():
        if tp + fn == 0:
            raise ValueError(f"evaluation_dataset_missing_positive:{key}")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        metrics[key] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                        "negative_frames": negative_frames[key]}
    mask_coverage = summarize_mask_coverage(mask_rows)
    report = {
        "frames": frames,
        "annotation_sha256": sha256_file(labels_path),
        "thresholds": {"min_recall": min_recall, "min_precision": min_precision},
        "metrics": metrics,
        "model_artifacts": _model_artifacts(face, "face") + _model_artifacts(plate, "license_plate"),
        "mask_coverage": mask_coverage,
        "source_revision": _source_revision(),
    }
    report["passed"] = derive_passed(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate face and plate detector recall on annotated local frames")
    parser.add_argument("labels", type=Path, help="JSONL with image, faces, license_plates fields")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--dataset-id", required=True, help="Stable name/version of the held-out annotations")
    parser.add_argument("--output", type=Path, help="Write machine-readable report JSON")
    args = parser.parse_args(argv)
    report = evaluate(args.labels, min_recall=args.min_recall, min_precision=args.min_precision)
    report["dataset_id"] = args.dataset_id
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
