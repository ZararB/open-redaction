#!/usr/bin/env python3
"""
Local human-in-the-loop labeling studio (bbox editor on contributor batches).

Local labeling studio: video canvas overlays, keyboard shortcuts,
save to labels.jsonl. Static UI + stdlib HTTP (no extra deps).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.network.batch_schema import (  # noqa: E402
    load_labels_jsonl,
    load_manifest,
    pseudo_labels_path,
    labels_path,
    write_labels_jsonl,
)


class StudioState:
    def __init__(self, batch_dir: Path):
        self.batch_dir = batch_dir.resolve()
        self.frames_dir = self.batch_dir / "frames"


STATE: StudioState | None = None


def _frame_names() -> list[str]:
    assert STATE is not None
    return sorted(p.name for p in STATE.frames_dir.glob("*.png"))


def _labels_file() -> Path:
    assert STATE is not None
    human = labels_path(STATE.batch_dir)
    if human.is_file():
        return human
    pseudo = pseudo_labels_path(STATE.batch_dir)
    return pseudo


def _row_for_frame(name: str) -> dict:
    rel = f"frames/{name}"
    rows = load_labels_jsonl(_labels_file())
    if rel in rows:
        return rows[rel]
    return {"image": rel, "faces": [], "license_plates": [], "note": "studio_new"}


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenRedactionStudio/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            return self._send_file(STATIC / "index.html")
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            return self._send_file(STATIC / rel)
        if path == "/api/meta":
            assert STATE is not None
            manifest = load_manifest(STATE.batch_dir)
            return self._send_json(
                200,
                {
                    "batch_id": manifest.batch_id,
                    "title": manifest.title,
                    "frame_count": len(_frame_names()),
                    "labels_file": str(_labels_file().name),
                },
            )
        if path == "/api/frames":
            return self._send_json(200, {"frames": _frame_names()})
        m = re.match(r"^/api/frame/([^/]+)$", path)
        if m:
            name = unquote(m.group(1))
            assert STATE is not None
            return self._send_file(STATE.frames_dir / name)
        m = re.match(r"^/api/labels/([^/]+)$", path)
        if m:
            name = unquote(m.group(1))
            return self._send_json(200, _row_for_frame(name))
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "invalid_json"})

        if parsed.path == "/api/save":
            assert STATE is not None
            frame = str(body.get("frame") or "")
            if not frame:
                return self._send_json(400, {"error": "frame_required"})
            rel = f"frames/{frame}"
            rows = load_labels_jsonl(_labels_file())
            rows[rel] = {
                "image": rel,
                "faces": body.get("faces") or [],
                "license_plates": body.get("license_plates") or [],
                "note": body.get("note") or "human_reviewed",
            }
            dest = labels_path(STATE.batch_dir)
            write_labels_jsonl(dest, rows)
            return self._send_json(200, {"saved": str(dest), "frame": frame})

        if parsed.path == "/api/save_all":
            assert STATE is not None
            frames = body.get("frames")
            if not isinstance(frames, dict):
                return self._send_json(400, {"error": "frames_dict_required"})
            rows = load_labels_jsonl(_labels_file())
            for frame, row in frames.items():
                rel = f"frames/{frame}"
                rows[rel] = {
                    "image": rel,
                    "faces": row.get("faces") or [],
                    "license_plates": row.get("license_plates") or [],
                    "note": row.get("note") or "human_reviewed",
                }
            dest = labels_path(STATE.batch_dir)
            write_labels_jsonl(dest, rows)
            return self._send_json(200, {"saved": str(dest), "count": len(frames)})

        self.send_error(404)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open Redaction labeling studio")
    parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Contributor batch dir (datasets/contributions/batch_…)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    global STATE
    batch = args.batch.resolve()
    if not (batch / "frames").is_dir():
        print(f"missing frames dir: {batch / 'frames'}", file=sys.stderr)
        return 1
    STATE = StudioState(batch)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open Redaction studio: http://{args.host}:{args.port}/")
    print(f"Batch: {batch}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
