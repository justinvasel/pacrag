from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "BACKGROUND": re.compile(r"^\s*BACKGROUND\s*$", re.MULTILINE),
    "ANALYSIS": re.compile(r"^\s*ANALYSIS\s*$", re.MULTILINE),
    "CONCLUSION": re.compile(r"^\s*CONCLUSION\s*$", re.MULTILINE),
    "DISCUSSION": re.compile(r"^\s*DISCUSSION\s*$", re.MULTILINE),
    "OPINION": re.compile(r"^\s*OPINION\s*$", re.MULTILINE),
}

_SIGNATURE_RE = re.compile(
    r"\n\s*(?:[A-Z][a-z]*\.?\s+){1,4}[A-Z][a-z]+\s*\n\s*Public Access Counselor",
)

_PAGE_MARKER_RE = re.compile(r"\s*Page\s+\d+\s+of\s+\d+\s*$")


def _find_section_headers(text: str) -> list[tuple[str, int, int]]:
    """Find all section headers and their positions.

    Returns list of (section_name, header_start, content_start) sorted by position.
    content_start points to the first character after the header line.
    """
    found: list[tuple[str, int, int]] = []
    for name, pattern in _SECTION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            header_start = match.start()
            header_end = match.end()
            has_newline = header_end < len(text) and text[header_end] == "\n"
            content_start = header_end + 1 if has_newline else header_end
            found.append((name, header_start, content_start))
    found.sort(key=lambda x: x[1])
    return found


def _trim_signature(text: str) -> str:
    """Remove trailing signature block and page markers from section text."""
    result = text
    result = _PAGE_MARKER_RE.sub("", result)
    sig_match = _SIGNATURE_RE.search(result)
    if sig_match:
        result = result[: sig_match.start()]
    return result.rstrip()


def detect_sections(
    full_text: str,
    opinion_id: str = "",
) -> dict[str, tuple[int, int]]:
    """Detect section boundaries in opinion text.

    Returns a dict mapping section names to (start, end) character offsets
    into full_text. Start is the first char after the section header line.
    End is the char before the next section header (or trimmed end of text).

    Returns empty dict if no standard sections are found.
    """
    if not full_text.strip():
        return {}

    headers = _find_section_headers(full_text)
    if not headers:
        if opinion_id:
            log.warning("no_sections_detected", opinion_id=opinion_id)
        return {}

    sections: dict[str, tuple[int, int]] = {}
    for i, (name, _header_start, content_start) in enumerate(headers):
        content_end = headers[i + 1][1] if i + 1 < len(headers) else len(full_text)

        raw_content = full_text[content_start:content_end]

        if i == len(headers) - 1:
            trimmed = _trim_signature(raw_content)
            actual_end = content_start + len(trimmed)
        else:
            trimmed = raw_content.rstrip()
            actual_end = content_start + len(trimmed)

        sections[name] = (content_start, actual_end)

    return sections
