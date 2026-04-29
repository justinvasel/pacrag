from __future__ import annotations

import re
from urllib.parse import urljoin

import httpx
import structlog
from selectolax.parser import HTMLParser

from pac_crawler.constants import (
    ADVISORY_INDEX_URL,
    BASE_URL,
    INFORMAL_INDEX_FALLBACK,
    INFORMAL_INDEX_URL,
)
from pac_crawler.http_client import RateLimiter, fetch_page
from pac_crawler.models import ListingEntry, OpinionKind, parse_opinion_id

log = structlog.get_logger()

_YEAR_RE = re.compile(r"(\d{4})")


def parse_year_links(html: str, base_url: str) -> list[tuple[str, int]]:
    """Extract year-page URLs from an index page. Returns (absolute_url, year) pairs."""
    tree = HTMLParser(html)
    results: list[tuple[str, int]] = []

    for node in tree.css("a"):
        href = node.attributes.get("href", "")
        if not href:
            continue

        text = node.text(strip=True)
        combined = f"{href} {text}"
        year_match = _YEAR_RE.search(combined)
        if year_match is None:
            continue

        year = int(year_match.group(1))
        if year < 1990 or year > 2100:
            continue

        if "advisory-opinion" in href or "informal-opinion" in href:
            absolute = urljoin(base_url, href)
            results.append((absolute, year))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def parse_opinion_links(
    html: str,
    listing_url: str,
    year: int,
    kind: OpinionKind,
) -> list[ListingEntry]:
    """Parse opinion entries from a year listing page."""
    tree = HTMLParser(html)
    entries: list[ListingEntry] = []

    for node in tree.css("a"):
        href = node.attributes.get("href", "")
        if not href or not href.lower().endswith(".pdf"):
            continue

        text = node.text(strip=True)
        if not text:
            continue

        parsed = parse_opinion_id(text)
        if parsed is None:
            continue

        opinion_id, _parsed_year, parsed_kind, sub_letter, is_revised = parsed

        if parsed_kind != kind:
            log.warning(
                "kind_mismatch",
                expected=kind.value,
                got=parsed_kind.value,
                opinion_id=opinion_id,
                listing_url=listing_url,
            )

        agency_name = _extract_agency_name(text, opinion_id)
        pdf_url = urljoin(BASE_URL, href)

        entry = ListingEntry(
            opinion_id=opinion_id,
            year=year,
            kind=kind,
            sub_letter=sub_letter,
            is_revised=is_revised,
            raw_title=text,
            agency_name=agency_name,
            pdf_url=pdf_url,
            listing_url=listing_url,
        )
        entries.append(entry)

    if not entries:
        log.warning("no_entries_found", listing_url=listing_url, year=year, kind=kind.value)

    return entries


def _extract_agency_name(raw_title: str, opinion_id: str) -> str:
    """Extract description/agency portion from title text after the opinion ID and colon."""
    colon_idx = raw_title.find(":")
    if colon_idx == -1:
        return raw_title
    return raw_title[colon_idx + 1 :].strip()


async def discover_all_listings(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    kind_filter: OpinionKind | None = None,
    year_filter: int | None = None,
) -> list[ListingEntry]:
    """Discover all opinion listings from index pages."""
    all_entries: list[ListingEntry] = []

    sources: list[tuple[str, str | None, OpinionKind]] = []
    if kind_filter is None or kind_filter == OpinionKind.FC:
        sources.append((ADVISORY_INDEX_URL, None, OpinionKind.FC))
    if kind_filter is None or kind_filter == OpinionKind.INF:
        sources.append((INFORMAL_INDEX_URL, INFORMAL_INDEX_FALLBACK, OpinionKind.INF))

    for index_url, fallback_url, kind in sources:
        year_links = await _fetch_year_links(client, limiter, index_url, fallback_url)

        if year_filter is not None:
            year_links = [(url, y) for url, y in year_links if y == year_filter]

        log.info(
            "discovered_year_pages",
            kind=kind.value,
            count=len(year_links),
            years=[y for _, y in year_links],
        )

        for page_url, year in year_links:
            entries = await _fetch_and_parse_year_page(client, limiter, page_url, year, kind)
            all_entries.extend(entries)

    log.info("discovery_complete", total_entries=len(all_entries))
    return all_entries


async def _fetch_year_links(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    primary_url: str,
    fallback_url: str | None,
) -> list[tuple[str, int]]:
    try:
        html = await fetch_page(client, primary_url, limiter)
        links = parse_year_links(html, primary_url)
        if links:
            return links
    except httpx.HTTPStatusError as e:
        log.warning("index_page_error", url=primary_url, status=e.response.status_code)

    if fallback_url:
        try:
            html = await fetch_page(client, fallback_url, limiter)
            return parse_year_links(html, fallback_url)
        except httpx.HTTPStatusError as e:
            log.warning("fallback_page_error", url=fallback_url, status=e.response.status_code)

    return []


async def _fetch_and_parse_year_page(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    url: str,
    year: int,
    kind: OpinionKind,
) -> list[ListingEntry]:
    try:
        html = await fetch_page(client, url, limiter)
        return parse_opinion_links(html, url, year, kind)
    except httpx.HTTPStatusError as e:
        log.error("year_page_error", url=url, status=e.response.status_code)
        return []
