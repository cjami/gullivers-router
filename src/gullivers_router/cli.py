"""Command-line entry point for Gulliver's Router."""

from __future__ import annotations

import argparse

from gullivers_router import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gullivers-router",
        description="Route queries between a local model and a cloud model by difficulty.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    print("Gulliver's Router is not implemented yet.")
    return 0
