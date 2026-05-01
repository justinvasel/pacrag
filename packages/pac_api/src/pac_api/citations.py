from __future__ import annotations

import re

import structlog

from pac_api.schemas import CitationRef

log = structlog.get_logger()

_CITATION_RE = re.compile(r"\((\d{2}-(?:FC|INF)-\d+[AB]?),\s*p\.\s*(\d+)\)")


def parse_citations(text: str) -> list[CitationRef]:
    return [
        CitationRef(opinion_id=m.group(1), page_num=int(m.group(2)))
        for m in _CITATION_RE.finditer(text)
    ]


def validate_citations(
    citations: list[CitationRef],
    retrieved_ids: frozenset[str],
) -> tuple[list[CitationRef], list[CitationRef]]:
    valid: list[CitationRef] = []
    invalid: list[CitationRef] = []
    for c in citations:
        if c.opinion_id in retrieved_ids:
            valid.append(c)
        else:
            invalid.append(CitationRef(opinion_id=c.opinion_id, page_num=c.page_num, valid=False))
    return valid, invalid


def check_response(
    text: str,
    retrieved_ids: frozenset[str],
) -> tuple[list[CitationRef], bool]:
    parsed = parse_citations(text)
    if not parsed:
        return [], True

    valid, invalid = validate_citations(parsed, retrieved_ids)

    if invalid:
        log.warning(
            "invalid_citations_detected",
            invalid_ids=[c.opinion_id for c in invalid],
            total_citations=len(parsed),
        )

    all_citations = valid + invalid
    return all_citations, len(invalid) == 0
