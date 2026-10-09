"""CLI entrypoint for the open-redaction package."""

from __future__ import annotations


def main() -> int:
    from agents.ml.face_blur.privacy_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
