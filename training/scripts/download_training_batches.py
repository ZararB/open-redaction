#!/usr/bin/env python3
"""Download public clips and export contributor labeling batches (PNG frames)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CATALOG = ROOT / "training" / "catalog" / "public_sources.json"
CONTRIB = ROOT / "datasets" / "contributions"
CACHE = CONTRIB / ".cache"
UA = "OpenRedactionTraining/1.0 (https://github.com/open-redaction; contributor batches)"


def _load_catalog() -> list[dict]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    return list(payload.get("samples") or [])


def _download_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180, headers={"User-Agent": UA}) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(1024 * 256):
                if chunk:
                    handle.write(chunk)


def _download_youtube(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format",
        "mp4",
        "-o",
        str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)


def _extract_frames(
    video: Path,
    out_dir: Path,
    *,
    start_sec: float,
    every_sec: float,
    max_frames: int,
    prefix: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
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
    duration = float(probe.stdout.strip() or "0")
    times: list[float] = []
    t = max(0.0, start_sec)
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


def _batch_dir_name(sample_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", sample_id.lower()).strip("_")[:48]
    return f"batch_{slug}"


def build_batch(sample: dict, *, skip_youtube: bool) -> Path | None:
    sample_id = str(sample["id"])
    if sample.get("requires_ytdlp") and skip_youtube:
        print(f"SKIP {sample_id} (YouTube; pass --include-youtube)")
        return None
    batch_cfg = sample.get("batch") or {}
    batch_dir = CONTRIB / _batch_dir_name(sample_id)
    frames_dir = batch_dir / "frames"
    if (batch_dir / "manifest.json").is_file() and any(frames_dir.glob("*.png")):
        print(f"SKIP existing {batch_dir.name}")
        return batch_dir

    url = str(sample["url"])
    cache_video = CACHE / f"{sample_id}.mp4"
    if "youtube.com" in url or "youtu.be" in url:
        _download_youtube(url, cache_video)
    else:
        ext = ".webm" if ".webm" in url.lower() else ".mp4"
        cache_video = CACHE / f"{sample_id}{ext}"
        if not cache_video.is_file():
            _download_http(url, cache_video)

    pngs = _extract_frames(
        cache_video,
        frames_dir,
        start_sec=float(batch_cfg.get("start_sec") or 0),
        every_sec=float(batch_cfg.get("every_sec") or 0.5),
        max_frames=int(batch_cfg.get("max_frames") or 60),
        prefix=str(batch_cfg.get("prefix") or "frame"),
    )

    from training.network.batch_schema import BatchManifest, save_manifest

    manifest = BatchManifest(
        batch_id=batch_dir.name,
        source_id=sample_id,
        title=str(sample.get("title") or sample_id),
        source_url=url,
        license_note=str(sample.get("license") or "unknown"),
        tags=list(sample.get("tags") or []),
        frame_count=len(pngs),
    )
    save_manifest(batch_dir, manifest)
    print(f"OK {batch_dir.name} frames={len(pngs)}")
    return batch_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", action="append", help="Only these catalog ids")
    parser.add_argument("--all-commons", action="store_true", help="All non-YouTube samples")
    parser.add_argument("--include-youtube", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    samples = _load_catalog()
    if args.list:
        for s in samples:
            print(s["id"], s.get("title"))
        return 0

    selected = samples
    if args.sample_id:
        ids = set(args.sample_id)
        selected = [s for s in samples if s["id"] in ids]
    elif args.all_commons:
        selected = [s for s in samples if not s.get("requires_ytdlp")]

    if not selected:
        parser.error("Pick --sample-id, --all-commons, or --include-youtube with ids")

    CONTRIB.mkdir(parents=True, exist_ok=True)
    for sample in selected:
        build_batch(sample, skip_youtube=not args.include_youtube)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
