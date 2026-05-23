"""postlens CLI entry point (v0.1.0 minimal driver)."""

from __future__ import annotations

import argparse
import sys

from postlens import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="postlens",
        description="Post-Transformer agent latency framework",
    )
    parser.add_argument("--version", action="version", version=f"postlens {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print package info")
    args = parser.parse_args(argv)

    if args.cmd == "info":
        print(f"postlens v{__version__}")
        print("Post-Transformer agent latency framework (alpha)")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
