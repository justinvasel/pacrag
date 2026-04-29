from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from pac_pipeline.parse import parse_all

log = structlog.get_logger()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pac-parse",
        description="Parse PAC opinion PDFs into structured JSON.",
    )
    parser.add_argument("--opinion-id", type=str, default=None, help="Parse a single opinion")
    parser.add_argument("--year", type=int, default=None, help="Parse opinions from this year")
    parser.add_argument(
        "--reparse", action="store_true", help="Force re-parse of already parsed opinions"
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")
    parser.add_argument("--db-path", type=Path, default=None, help="DuckDB path")
    parser.add_argument("--lit-binary", type=str, default=None, help="Path to lit binary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    data_dir = args.data_dir or Path(os.environ.get("PAC_DATA_DIR", "./data"))
    db_path = args.db_path or Path(os.environ.get("PAC_DUCKDB_PATH", "./data/catalog/pac.duckdb"))
    lit_binary = args.lit_binary or os.environ.get("PAC_LIT_BINARY", "lit")

    parsed, errors = parse_all(
        db_path=db_path,
        data_dir=data_dir,
        year=args.year,
        opinion_id=args.opinion_id,
        reparse=args.reparse,
        lit_binary=lit_binary,
    )

    log.info("parse_complete", parsed=parsed, errors=errors)
