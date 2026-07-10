"""Command-line entry point for Gulliver's Router."""

from __future__ import annotations

import argparse
from pathlib import Path

from gullivers_router import __version__, practice, router, training


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
    _add_run_parser(subparsers)
    _add_score_practice_parser(subparsers)
    _add_train_parser(subparsers)
    _add_train_router_parser(subparsers)
    return parser


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser("run", help="Run the batch router.")
    run_parser.add_argument(
        "--input",
        type=Path,
        default=router.DEFAULT_INPUT,
        help="Tasks JSON file.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=router.DEFAULT_OUTPUT,
        help="Path for results JSON.",
    )
    run_parser.add_argument(
        "--router-weights",
        type=Path,
        default=router.DEFAULT_ROUTER_WEIGHTS,
        help="Exported numpy router weights.",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=training.DEFAULT_CONCURRENCY,
        help="Concurrent serverless requests for cloud-routed tasks.",
    )
    run_parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Write route diagnostics without generating answers.",
    )
    run_parser.add_argument(
        "--local-cascade",
        action="store_true",
        help="Let local-routed borderline tasks self-check before accepting local answers.",
    )
    run_parser.add_argument(
        "--cascade-margin",
        type=float,
        default=router.DEFAULT_CASCADE_MARGIN,
        help="Risk margin below the routing threshold that still gets local cascade checks.",
    )


def _add_score_practice_parser(subparsers: argparse._SubParsersAction) -> None:
    score_parser = subparsers.add_parser(
        "score-practice",
        help="Score practice task answers against the reference answer set with the judge model.",
    )
    score_parser.add_argument(
        "--tasks",
        type=Path,
        default=practice.DEFAULT_INPUT,
        help="Practice tasks JSON file.",
    )
    score_parser.add_argument(
        "--results",
        type=Path,
        default=practice.DEFAULT_RESULTS,
        help="Router results JSON file to score.",
    )
    score_parser.add_argument(
        "--answer-set",
        type=Path,
        default=practice.DEFAULT_ANSWER_SET,
        help="Reference answer set JSON file.",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        default=practice.DEFAULT_SCORE_OUTPUT,
        help="Path for the scoring report JSON.",
    )
    score_parser.add_argument(
        "--workers",
        type=int,
        default=training.DEFAULT_CONCURRENCY,
        help="Concurrent judge requests.",
    )


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
        help="Fraction of prompts held out and split evenly into calibration and test sets.",
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
        help="Minimum acceptable routed answer quality; 3.0 corresponds to 'adequate'.",
    )
    router_parser.add_argument(
        "--accuracy-gate",
        type=float,
        default=training.DEFAULT_ACCURACY_GATE,
        help="Portfolio accuracy floor the blended pass rate must clear (e.g. 0.80).",
    )
    router_parser.add_argument(
        "--margin",
        type=float,
        default=training.DEFAULT_TARGET_MARGIN,
        help="Safety buffer added to the gate when picking thresholds, against calibration drift.",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        router.run(
            input_path=args.input,
            output_path=args.output,
            router_weights=args.router_weights,
            workers=args.workers,
            classify_only=args.classify_only,
            local_cascade=args.local_cascade,
            cascade_margin=args.cascade_margin,
        )
    elif args.command == "score-practice":
        practice.score_practice(
            tasks_path=args.tasks,
            results_path=args.results,
            answer_set_path=args.answer_set,
            output_path=args.output,
            workers=args.workers,
        )
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
            accuracy_gate=args.accuracy_gate,
            target_margin=args.margin,
        )
    else:
        parser.print_help()
    return 0
