"""Unified entrypoint for CLI and UI usage."""

from __future__ import annotations

import argparse
import sys

from sprite_recon.main import main as cli_main
from sprite_recon.ui.app import create_app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprite reconstruction launcher")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the web UI")
    ui_parser.add_argument("--host", default="127.0.0.1", help="UI host")
    ui_parser.add_argument("--port", type=int, default=5000, help="UI port")
    ui_parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    cli_parser = subparsers.add_parser("cli", help="Run the CLI optimizer")
    cli_parser.add_argument("cli_args", nargs=argparse.REMAINDER, help="Arguments passed to the CLI")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command in (None, "ui"):
        app = create_app()
        app.run(host=args.host, port=args.port, debug=bool(args.debug))
        return

    if args.command == "cli":
        argv = [sys.argv[0], *args.cli_args]
        sys.argv = argv
        cli_main()
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
