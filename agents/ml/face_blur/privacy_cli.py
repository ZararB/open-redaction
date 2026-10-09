"""CLI for local privacy redaction before upload (faces + license plates)."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Redact faces and license plates in a video file (local ONNX/OpenCV stack)."
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--no-faces", action="store_true", help="Skip face redaction")
    parser.add_argument("--no-plates", action="store_true", help="Skip license-plate redaction")
    parser.add_argument("--blur-strength", type=int, default=51, help="Gaussian blur kernel (odd)")
    parser.add_argument("--json", action="store_true", help="Print result JSON to stdout")
    args = parser.parse_args(argv)

    blur_faces = not args.no_faces
    blur_plates = not args.no_plates
    if not blur_faces and not blur_plates:
        print("At least one of faces or plates must be enabled.", file=sys.stderr)
        return 2

    from agents.services.privacy_redaction_service import PrivacyRedactionService

    service = PrivacyRedactionService(
        blur_strength=args.blur_strength,
        blur_faces=blur_faces,
        blur_plates=blur_plates,
    )
    result = service.redact_video(args.input, args.output)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Wrote {result['output_path']} targets={result.get('targets')}")
        print("Review the output before upload; detector completion does not prove every face or plate was found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
