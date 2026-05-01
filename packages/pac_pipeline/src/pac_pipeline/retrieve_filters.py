from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

from pac_pipeline.extract_models import ISSUE_TAGS, ISSUE_TAGS_SET
from pac_pipeline.retrieve_models import InferredFilters

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

log = structlog.get_logger()

_TOOL_NAME = "infer_filters"

_SYSTEM_PROMPT = (
    """\
You are a search filter assistant for a database of Indiana Public Access Counselor (PAC) \
advisory opinions. Given a user's question, suggest metadata filters to narrow retrieval.

Only suggest filters when the question clearly implies them. If the question is broad or \
general, return no filters (empty values).

Field definitions:

- year_min / year_max: Year range if the question references a time period. Use null if \
not specified.
- agency_type: The type of government agency. Only set if the question clearly references \
a specific type. Values:
  "county", "city", "town", "school_corporation", "state_agency", "law_enforcement", \
"court", "library", "utility", "other"
- statute_areas: Which area of law is at issue. Values:
  "APRA" (Access to Public Records Act, IC 5-14-3) -- for public records, FOIA-like requests
  "ODL" (Open Door Law, IC 5-14-1.5) -- for open meetings, executive sessions, meeting notice
- finding: The counselor's conclusion, if the question implies one. Values:
  "violation", "no_violation", "mixed", "moot", "no_jurisdiction", "withdrawn"
- issue_tags: Select from this controlled list ONLY if the question directly relates:
  """
    + ", ".join(ISSUE_TAGS)
    + """

Important:
- Prefer fewer filters over more. Aggressive filtering can miss relevant results.
- Map common language to statute_areas: "public records" / "records request" / "FOIA" -> \
"APRA"; "open meetings" / "executive session" / "meeting notice" -> "ODL".
- Do not set finding unless the question explicitly asks about violations or outcomes.
"""
)

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "year_min": {"type": ["integer", "null"], "description": "Earliest year, or null"},
        "year_max": {"type": ["integer", "null"], "description": "Latest year, or null"},
        "agency_type": {
            "type": ["string", "null"],
            "enum": [
                "county",
                "city",
                "town",
                "school_corporation",
                "state_agency",
                "law_enforcement",
                "court",
                "library",
                "utility",
                "other",
            ],
        },
        "statute_areas": {
            "type": "array",
            "items": {"type": "string", "enum": ["APRA", "ODL"]},
            "default": [],
        },
        "finding": {
            "type": ["string", "null"],
            "enum": [
                "violation",
                "no_violation",
                "mixed",
                "moot",
                "no_jurisdiction",
                "withdrawn",
            ],
        },
        "issue_tags": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
    "required": [],
    "additionalProperties": False,
}

_TOOL_DEF = {
    "name": _TOOL_NAME,
    "description": "Suggest metadata filters to narrow search results for PAC opinions.",
    "input_schema": _TOOL_SCHEMA,
}


async def infer_filters(question: str, client: AsyncAnthropic) -> InferredFilters:
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL_DEF],  # type: ignore[list-item]
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": question}],
    )

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"),
        None,
    )

    if tool_block is None:
        log.warning("filter_inference_no_tool_use")
        return InferredFilters()

    raw = tool_block.input
    if not isinstance(raw, dict):
        return InferredFilters()

    raw_tags: list[str] = raw.get("issue_tags", [])  # type: ignore[assignment]
    valid_tags = [t for t in raw_tags if t in ISSUE_TAGS_SET]
    if len(valid_tags) != len(raw_tags):
        log.info(
            "filter_inference_stripped_tags",
            original=raw_tags,
            valid=valid_tags,
        )

    cleaned: dict[str, object] = {}
    for key in ("year_min", "year_max", "agency_type", "statute_areas", "finding"):
        val = raw.get(key)
        if val is not None:
            cleaned[key] = val
    cleaned["issue_tags"] = valid_tags

    return InferredFilters.model_validate(cleaned)


def to_qdrant_filter(filters: InferredFilters) -> Filter | None:
    if not filters.has_any():
        return None

    conditions: list[FieldCondition] = []

    if filters.year_min is not None or filters.year_max is not None:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=filters.year_min,
                    lte=filters.year_max,
                ),
            )
        )

    if filters.agency_type is not None:
        conditions.append(
            FieldCondition(key="agency_type", match=MatchValue(value=filters.agency_type))
        )

    if filters.statute_areas:
        conditions.append(
            FieldCondition(key="statute_areas", match=MatchAny(any=list(filters.statute_areas)))
        )

    if filters.finding is not None:
        conditions.append(FieldCondition(key="finding", match=MatchValue(value=filters.finding)))

    if filters.issue_tags:
        conditions.append(FieldCondition(key="issue_tags", match=MatchAny(any=filters.issue_tags)))

    return Filter(must=conditions)  # type: ignore[arg-type]
