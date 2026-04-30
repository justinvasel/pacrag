from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from pac_pipeline.extract import extract_all

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
        prog="pac-extract",
        description="Extract structured metadata from parsed PAC opinions via LLM.",
    )
    parser.add_argument("--opinion-id", type=str, default=None, help="Extract a single opinion")
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Force re-extraction of already extracted opinions",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of opinions to extract")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")
    parser.add_argument("--db-path", type=Path, default=None, help="DuckDB path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    data_dir = args.data_dir or Path(os.environ.get("PAC_DATA_DIR", "./data"))
    db_path = args.db_path or Path(os.environ.get("PAC_DUCKDB_PATH", "./data/catalog/pac.duckdb"))

    extracted, errors = extract_all(
        db_path=db_path,
        data_dir=data_dir,
        opinion_id=args.opinion_id,
        reextract=args.reextract,
        limit=args.limit,
    )

    log.info("extract_complete", extracted=extracted, errors=errors)
