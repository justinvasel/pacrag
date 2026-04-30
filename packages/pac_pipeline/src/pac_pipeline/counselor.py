from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

_KNOWN_COUNSELORS: frozenset[str] = frozenset(
    {
        "Luke H. Britt",
        "Joseph B. Hoage",
        "Karen Davis",
        "Andrew J. Kossack",
        "Michael A. Hurst",
        "Jennifer G. Ruby",
        "Heather Willis Neal",
        "Britt Prendergast",
    }
)

_SIGNATURE_RE = re.compile(
    r"\n\s*((?:[A-Z][a-z]*\.?\s+){1,4}[A-Z][a-z]+)\s*\n\s*Public Access Counselor",
)


def extract_counselor(text: str) -> str | None:
    match = _SIGNATURE_RE.search(text)
    if not match:
        return None

    name = " ".join(match.group(1).split())

    if name not in _KNOWN_COUNSELORS:
        log.warning("unknown_counselor_name", extracted_name=name)

    return name
