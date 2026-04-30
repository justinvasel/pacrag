from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from pac_pipeline.embed import embed_all

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
        prog="pac-embed",
        description="Chunk, embed, and index parsed PAC opinions.",
    )
    parser.add_argument("--opinion-id", type=str, default=None, help="Embed a single opinion")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-embedding of already embedded opinions",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of opinions to embed")
    parser.add_argument("--db-path", type=Path, default=None, help="DuckDB path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    db_path = args.db_path or Path(os.environ.get("PAC_DUCKDB_PATH", "./data/catalog/pac.duckdb"))

    embedded, errors = embed_all(
        db_path=db_path,
        opinion_id=args.opinion_id,
        reembed=args.reindex,
        limit=args.limit,
    )

    log.info("embed_complete", embedded=embedded, errors=errors)
