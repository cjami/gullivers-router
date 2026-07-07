"""Command-line entry point for Gulliver's Router."""

from __future__ import annotations

import argparse

from gullivers_router import __version__, router, training


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
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="Serve the runtime router.")
    subparsers.add_parser("train", help="Train the matrix-factorization router.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        router.run()
    elif args.command == "train":
        training.train()
    else:
        parser.print_help()
    return 0
