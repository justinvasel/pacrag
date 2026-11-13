from __future__ import annotations

import re

from pac_api.citations import parse_citations, validate_citations

_DISCLAIMER_RE = re.compile(
    r"advisory\s+and\s+persuasive\s+only",
    re.IGNORECASE,
)

_OVERCLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bthe\s+law\s+requires\b",
        r"\blegally\s+obligat",
        r"\bthis\s+is\s+binding\b",
        r"\byou\s+must\b",
        r"\byou\s+are\s+required\b",
        r"\byou\s+are\s+legally\b",
        r"\bagencies?\s+must\s+comply\b",
    ]
]

_PAC_QUALIFIER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bthe\s+PAC\s+(opined|found|concluded|determined|stated)\b",
        r"\baccording\s+to\s+(the\s+)?PAC\b",
        r"\bin\s+the\s+PAC['s]*\s+(opinion|view)\b",
        r"\bthe\s+counselor\s+(opined|found|concluded)\b",
    ]
]


def compute_citation_precision(
    answer_text: str,
    retrieved_opinion_ids: frozenset[str],
) -> tuple[float, int, int]:
    citations = parse_citations(answer_text)
    if not citations:
        return 1.0, 0, 0

    valid, _ = validate_citations(citations, retrieved_opinion_ids)
    precision = len(valid) / len(citations)
    return precision, len(citations), len(valid)


def compute_hedging_score(answer_text: str) -> tuple[bool, bool, float]:
    hedging_present = bool(_DISCLAIMER_RE.search(answer_text))

    has_qualifier = any(p.search(answer_text) for p in _PAC_QUALIFIER_PATTERNS)
    overclaim_detected = False
    if not has_qualifier:
        overclaim_detected = any(p.search(answer_text) for p in _OVERCLAIM_PATTERNS)

    if hedging_present and not overclaim_detected:
        score = 1.0
    elif hedging_present or not overclaim_detected:
        score = 0.5
    else:
        score = 0.0

    return hedging_present, overclaim_detected, score
