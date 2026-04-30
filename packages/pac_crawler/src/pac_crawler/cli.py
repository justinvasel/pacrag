from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from pac_crawler.downloader import download_opinion_pdf
from pac_crawler.http_client import RateLimiter, build_client
from pac_crawler.listing_parser import discover_all_listings
from pac_crawler.manifest import ManifestDB
from pac_crawler.models import ListingEntry, OpinionKind

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
        prog="pac-crawl",
        description="Crawl Indiana PAC advisory opinions and build the manifest.",
    )
    parser.add_argument("--year", type=int, default=None, help="Only crawl this year (e.g. 2025)")
    parser.add_argument(
        "--kind",
        choices=["FC", "INF", "all"],
        default="all",
        help="Opinion kind to crawl (default: all)",
    )
    parser.add_argument(
        "--listing-only",
        action="store_true",
        help="Only fetch listings, skip PDF downloads",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to download")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")
    parser.add_argument("--db-path", type=Path, default=None, help="DuckDB path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


async def run(
    data_dir: Path,
    db_path: Path,
    year_filter: int | None,
    kind_filter: OpinionKind | None,
    listing_only: bool,
    limit: int | None,
) -> None:
    with ManifestDB(db_path) as db:
        async with build_client() as client:
            limiter = RateLimiter()

            entries = await discover_all_listings(
                client, limiter, kind_filter=kind_filter, year_filter=year_filter
            )

            if not entries:
                log.warning("no_entries_discovered")
                return

            db.upsert_listings(entries)
            log.info("upserted_listings", count=len(entries))

            if year_filter is None:
                _mark_missing(db, entries, kind_filter)
            else:
                log.info("skipping_mark_missing", reason="year filter active")

            if listing_only:
                log.info("listing_only_mode", skipping_downloads=True)
            else:
                existing_hashes = db.get_existing_hashes()
                pending = [
                    e
                    for e in entries
                    if existing_hashes.get(e.opinion_id) is None
                    or existing_hashes[e.opinion_id] == ""
                ]

                if not pending:
                    log.info("no_pending_downloads")
                else:
                    to_download = pending[:limit] if limit else pending
                    ok_count = 0
                    skip_count = 0
                    err_count = 0

                    for entry in to_download:
                        existing_sha = existing_hashes.get(entry.opinion_id)
                        path, sha, status = await download_opinion_pdf(
                            client, entry, data_dir, existing_sha, limiter
                        )
                        if status in ("ok", "skipped"):
                            db.update_download(entry.opinion_id, str(path), sha, "ok")
                            if status == "ok":
                                ok_count += 1
                            else:
                                skip_count += 1
                        elif status == "http_error":
                            db.update_download(entry.opinion_id, "", "", "http_error")
                            err_count += 1

                    log.info(
                        "downloads_complete",
                        ok=ok_count,
                        skipped=skip_count,
                        errors=err_count,
                    )

        counts = db.count_by_status()
        log.info("manifest_summary", **counts)


def _mark_missing(
    db: ManifestDB,
    entries: list[ListingEntry],
    kind_filter: OpinionKind | None,
) -> None:
    """Mark opinions not seen in this crawl as missing."""
    fc_ids = {e.opinion_id for e in entries if e.kind == OpinionKind.FC}
    inf_ids = {e.opinion_id for e in entries if e.kind == OpinionKind.INF}

    if kind_filter is None or kind_filter == OpinionKind.FC:
        db.mark_missing(fc_ids, "FC")
    if kind_filter is None or kind_filter == OpinionKind.INF:
        db.mark_missing(inf_ids, "INF")


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    data_dir = args.data_dir or Path(os.environ.get("PAC_DATA_DIR", "./data"))
    db_path = args.db_path or Path(os.environ.get("PAC_DUCKDB_PATH", "./data/catalog/pac.duckdb"))

    kind_filter: OpinionKind | None = None
    if args.kind != "all":
        kind_filter = OpinionKind(args.kind)

    asyncio.run(
        run(
            data_dir=data_dir,
            db_path=db_path,
            year_filter=args.year,
            kind_filter=kind_filter,
            listing_only=args.listing_only,
            limit=args.limit,
        )
    )
