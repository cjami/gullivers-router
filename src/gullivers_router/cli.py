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
    _add_train_parser(subparsers)
    return parser


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser("train", help="Build the labelled training dataset.")
    train_parser.add_argument(
        "--samples",
        type=int,
        default=training.SAMPLES_PER_CATEGORY,
        help="Prompts per category; lower it to rehearse, raise it to scale up.",
    )
    train_parser.add_argument(
        "--out",
        default=training.DEFAULT_OUT,
        help="Directory for the pipeline's artifacts (use a distinct dir per run size).",
    )
    train_parser.add_argument(
        "--margin",
        type=int,
        default=training.DEFAULT_MARGIN,
        help="Cloud-minus-local score gap that labels a prompt for the cloud.",
    )
    train_parser.add_argument(
        "--stages",
        default=",".join(training.STAGES),
        help="Comma-separated stages to run (subset of: %(default)s).",
    )
    train_parser.add_argument(
        "--workers",
        type=int,
        default=training.DEFAULT_CONCURRENCY,
        help="Concurrent serverless requests for the cloud and judge stages.",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        router.run()
    elif args.command == "train":
        training.train(
            samples_per_category=args.samples,
            out=args.out,
            margin=args.margin,
            stages=tuple(stage.strip() for stage in args.stages.split(",") if stage.strip()),
            workers=args.workers,
        )
    else:
        parser.print_help()
    return 0
