"""Command-line entry point for Gulliver's Router."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    _add_train_router_parser(subparsers)
    return parser


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser("train", help="Build the router training dataset.")
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


def _add_train_router_parser(subparsers: argparse._SubParsersAction) -> None:
    router_parser = subparsers.add_parser(
        "train-router",
        help="Train the routing model from judge scores and embeddings.",
    )
    router_parser.add_argument(
        "--out",
        default=training.DEFAULT_OUT,
        help="Directory holding labels.jsonl judge rows and embeddings.jsonl.",
    )
    router_parser.add_argument(
        "--val-fraction",
        type=float,
        default=training.DEFAULT_VAL_FRACTION,
        help="Fraction of prompts held out for cost-quality evaluation.",
    )
    router_parser.add_argument(
        "--seed",
        type=int,
        default=training.DEFAULT_SEED,
        help="Random seed for the stratified split and cross-validation.",
    )
    router_parser.add_argument(
        "--quality-floor",
        type=float,
        default=training.DEFAULT_QUALITY_FLOOR,
        help="Minimum acceptable routed answer quality; 4.0 corresponds to 'good'.",
    )
    router_parser.add_argument(
        "--target-pass-rate",
        type=float,
        default=training.DEFAULT_TARGET_PASS_RATE,
        help="Required fraction of routed validation answers meeting the quality floor.",
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
            stages=tuple(stage.strip() for stage in args.stages.split(",") if stage.strip()),
            workers=args.workers,
        )
    elif args.command == "train-router":
        training.train_router(
            Path(args.out),
            val_fraction=args.val_fraction,
            seed=args.seed,
            quality_floor=args.quality_floor,
            target_pass_rate=args.target_pass_rate,
        )
    else:
        parser.print_help()
    return 0
