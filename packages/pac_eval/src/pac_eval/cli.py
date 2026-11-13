from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv


def _configure_logging(verbose: bool) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pac-eval",
        description="Evaluate PAC RAG retrieval and generation quality",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the eval gold set")
    run_parser.add_argument("--gold-set", type=Path, default=None, help="Path to gold set YAML")
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval"),
        help="Directory for Parquet output",
    )
    run_parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only compute retrieval metrics",
    )
    run_parser.add_argument(
        "--with-filters",
        action="store_true",
        help="Apply statute_area from gold entries as filters",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max parallel questions (default: 3)",
    )
    run_parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    diff_parser = sub.add_parser("diff", help="Compare two eval runs")
    diff_parser.add_argument("path_a", type=Path, help="First scorecard Parquet")
    diff_parser.add_argument("path_b", type=Path, help="Second scorecard Parquet")
    diff_parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Regression threshold (default: 0.01)",
    )
    diff_parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def _run_eval(args: argparse.Namespace) -> None:
    from pac_eval.gold_set import load_gold_set
    from pac_eval.runner import run_eval
    from pac_eval.scorecard import write_scorecard

    log = structlog.get_logger()

    gold_set = load_gold_set(args.gold_set)
    log.info(
        "gold_set_loaded",
        version=gold_set.version,
        entries=len(gold_set.entries),
    )

    scorecard = asyncio.run(
        run_eval(
            gold_set,
            skip_generation=args.skip_generation,
            with_filters=args.with_filters,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )

    path = write_scorecard(scorecard, args.output_dir)
    log.info("scorecard_written", path=str(path))

    s = scorecard.summary
    print(f"\nEval Results ({scorecard.run_id})")
    print(f"  Questions:          {s.questions_evaluated}")
    print(f"  Errors:             {s.questions_with_errors}")
    print(f"  Recall@8:           {s.mean_recall_at_8:.3f}")
    print(f"  Recall@16:          {s.mean_recall_at_16:.3f}")
    print(f"  MRR:                {s.mean_mrr:.3f}")
    if not args.skip_generation:
        print(f"  Citation Precision: {s.mean_citation_precision:.3f}")
        print(f"  Fact Coverage:      {s.mean_fact_coverage:.3f}")
        print(f"  Hedging Score:      {s.mean_hedging_score:.3f}")
        print(f"  Hedging Rate:       {s.hedging_rate:.3f}")
    print(f"\n  Scorecard: {path}")


def _run_diff(args: argparse.Namespace) -> None:
    from pac_eval.diff import diff_scorecards
    from pac_eval.scorecard import read_scorecard

    a = read_scorecard(args.path_a)
    b = read_scorecard(args.path_b)
    report = diff_scorecards(a, b, threshold=args.threshold)
    print(report)


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(getattr(args, "verbose", False))

    if args.command == "run":
        _run_eval(args)
    elif args.command == "diff":
        _run_diff(args)
    else:
        parser.print_help()
        sys.exit(1)
