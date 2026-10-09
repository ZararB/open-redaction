from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], here.parents[2], here.parents[3]):
        if (candidate / "datasets" / "privacy_redaction").is_dir():
            return candidate
    return here.parents[2]


REPO_ROOT = _repo_root()
SCRIPTS = REPO_ROOT / "agents" / "ml" / "face_blur" / "scripts"
TOY_DIR = REPO_ROOT / "datasets" / "privacy_redaction" / "toy_ci"


class MirrorDetector:
    """Test helper: predictions equal annotation boxes (see evaluation._detector_predictions)."""

    name = "mirror_test_detector"

    def detect_for_annotations(self, frame, truth_boxes):
        _ = frame
        return [tuple(int(v) for v in box) for box in truth_boxes]


def test_generate_toy_dataset_script(tmp_path):
    out = tmp_path / "toy"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_toy_eval_dataset.py"), "--out", str(out), "--frames", "100"],
        check=True,
        cwd=str(REPO_ROOT),
    )
    labels = out / "labels.jsonl"
    assert labels.is_file()
    lines = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 100
    assert (out / "frames" / "frame_0000.png").is_file()


def test_toy_labels_schema_on_disk():
    labels = TOY_DIR / "labels.jsonl"
    if not labels.is_file():
        pytest.skip("toy_ci not generated yet; run generate_toy_eval_dataset.py")
    for line in labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "image" in row and "faces" in row and "license_plates" in row
        img = TOY_DIR / row["image"]
        assert img.is_file(), img


def test_evaluation_passes_on_toy_with_mirror_detectors(monkeypatch):
    labels = TOY_DIR / "labels.jsonl"
    if not labels.is_file():
        pytest.skip("toy_ci not generated yet")

    from agents.ml.face_blur import evaluation as evaluation_mod
    from agents.ml.face_blur.evaluation import evaluate
    from agents.ml.face_blur.release_registry import validate_report

    def _perfect_masks(*_args, **_kwargs):
        return {"face_misses": [], "plate_misses": [], "encoded_frames": 1}

    monkeypatch.setattr(evaluation_mod, "evaluate_encoded_frame_masks", _perfect_masks)

    mirror = MirrorDetector()
    report = evaluate(
        labels,
        min_recall=0.95,
        min_precision=0.90,
        face_detector=mirror,
        plate_detector=mirror,
    )
    report["dataset_id"] = "toy-ci-v1"
    assert report["passed"] is True
    assert report["frames"] >= 100
    validate_report(report, labels_path=labels)


def test_build_eval_frames_local_video(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=5",
            "-t",
            "2",
            str(video),
        ],
        check=True,
    )
    out = tmp_path / "extracted"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_eval_frames.py"),
            "--video",
            str(video),
            "--out",
            str(out),
            "--every-sec",
            "0.5",
            "--max-frames",
            "3",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    labels = out / "labels.jsonl"
    assert labels.is_file()
    frames = list((out / "frames").glob("*.png"))
    assert len(frames) == 3
