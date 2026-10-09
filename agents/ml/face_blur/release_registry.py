"""Privacy redaction release registration gates (framework-agnostic)."""
from __future__ import annotations

import math
from pathlib import Path

from .evaluation import derive_passed
from .manifest import sha256_file


def validate_report(report: dict, *, labels_path: Path | None = None) -> None:
    if not report.get("dataset_id"):
        raise ValueError("evaluation_report_dataset_id_required")
    if derive_passed(report) is not True:
        raise ValueError("evaluation_report_did_not_derive_pass")
    mask = report.get("mask_coverage") or {}
    if mask.get("passed") is not True:
        raise ValueError("encoded_mask_coverage_required")
    if int(report.get("frames") or 0) < 100:
        raise ValueError("evaluation_report_needs_at_least_100_frames")
    annotation_digest = str(report.get("annotation_sha256") or "")
    if len(annotation_digest) != 64 or any(c not in "0123456789abcdef" for c in annotation_digest):
        raise ValueError("annotation_digest_required")
    thresholds = report.get("thresholds") or {}
    min_recall = float(thresholds.get("min_recall") or 0)
    min_precision = float(thresholds.get("min_precision") or 0)
    if not all(math.isfinite(v) for v in (min_recall, min_precision)) or min_recall < 0.95 or min_precision < 0.90:
        raise ValueError("evaluation_thresholds_too_low")
    metrics = report.get("metrics") or {}
    for target in ("face", "license_plate"):
        item = metrics.get(target) or {}
        tp, fp, fn = (int(item.get(key) or 0) for key in ("tp", "fp", "fn"))
        negatives = int(item.get("negative_frames") or 0)
        if tp + fn < 50 or not 10 <= negatives <= int(report["frames"]) or min(tp, fp, fn) < 0:
            raise ValueError(f"invalid_evaluation_metrics:{target}")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        observed_precision = float(item.get("precision", -1))
        observed_recall = float(item.get("recall", -1))
        if (
            not all(math.isfinite(v) for v in (observed_precision, observed_recall))
            or abs(observed_precision - precision) > 1e-6
            or abs(observed_recall - recall) > 1e-6
        ):
            raise ValueError(f"inconsistent_evaluation_metrics:{target}")
        if precision < min_precision or recall < min_recall:
            raise ValueError(f"evaluation_target_below_threshold:{target}")
    if labels_path is not None:
        if not labels_path.is_file():
            raise ValueError("annotation_labels_missing")
        digest = sha256_file(labels_path)
        if digest != str(report.get("annotation_sha256") or ""):
            raise ValueError("annotation_sha256_mismatch")
