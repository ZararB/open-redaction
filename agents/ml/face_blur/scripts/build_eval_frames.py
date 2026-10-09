"""Extract PNG frames from local video or bundled redaction test samples."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SAMPLES = _REPO_ROOT / "samples" / "redaction_test_samples.json"
UA = "OpenRedactionEval/1.0 (https://github.com/open-redaction; privacy dataset tooling)"


def load_samples(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(s["id"]): s for s in payload.get("samples") or [] if s.get("id")}


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA}) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(1024 * 256):
                if chunk:
                    handle.write(chunk)


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    every_sec: float,
    max_frames: int,
    prefix: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(duration_probe.stdout.strip() or "0")
    if duration <= 0:
        raise ValueError(f"invalid_video_duration:{video}")

    times: list[float] = []
    t = 0.0
    while t < duration and len(times) < max_frames:
        times.append(t)
        t += every_sec

    written: list[Path] = []
    for i, ts in enumerate(times):
        out_png = out_dir / f"{prefix}_{i:04d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(out_png),
            ],
            check=True,
        )
        written.append(out_png)
    return written


def write_labels_jsonl(frame_paths: list[Path], labels_path: Path, *, rel_root: Path) -> None:
    lines = []
    for png in frame_paths:
        rel = png.relative_to(rel_root)
        lines.append(
            json.dumps(
                {
                    "image": str(rel).replace("\\", "/"),
                    "faces": [],
                    "license_plates": [],
                    "note": "unlabeled — fill boxes before running evaluation gate",
                }
            )
        )
    labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build unlabeled evaluation frame exports")
    parser.add_argument("--video", type=Path, help="Local video file")
    parser.add_argument("--url", help="Direct video URL to download")
    parser.add_argument("--sample-id", help="Id from face_redaction_test_samples.json")
    parser.add_argument("--samples-json", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "datasets" / "privacy_redaction" / "extracted",
    )
    parser.add_argument("--every-sec", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--prefix", default="frame")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "_cache"
    cache.mkdir(exist_ok=True)

    source_video: Path | None = args.video
    if args.sample_id:
        samples = load_samples(args.samples_json)
        sample = samples.get(args.sample_id)
        if not sample:
            raise SystemExit(f"unknown sample id: {args.sample_id}")
        url = str(sample["url"])
        ext = Path(urlparse(url).path).suffix or ".webm"
        source_video = cache / f"{args.sample_id}{ext}"
        if not source_video.is_file():
            download_url(url, source_video)
    elif args.url:
        ext = Path(urlparse(args.url).path).suffix or ".mp4"
        source_video = cache / f"download{ext}"
        if not source_video.is_file():
            download_url(args.url, source_video)

    if source_video is None or not source_video.is_file():
        print("Provide --video, --url, or --sample-id", file=sys.stderr)
        return 2

    frames_dir = out / "frames"
    paths = extract_frames(
        source_video,
        frames_dir,
        every_sec=args.every_sec,
        max_frames=args.max_frames,
        prefix=args.prefix,
    )
    labels = out / "labels.jsonl"
    write_labels_jsonl(paths, labels, rel_root=out)
    print(json.dumps({"frames": len(paths), "labels": str(labels), "video": str(source_video)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
