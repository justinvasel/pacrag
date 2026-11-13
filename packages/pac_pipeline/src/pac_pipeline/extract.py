from __future__ import annotations

from datetime import date
from pathlib import Path

import structlog
from anthropic import Anthropic
from tqdm import tqdm

from pac_pipeline.config import settings
from pac_pipeline.counselor import extract_counselor
from pac_pipeline.extract_db import ExtractDB
from pac_pipeline.extract_models import (
    EXTRACTOR_VERSION,
    ISSUE_TAGS,
    ISSUE_TAGS_SET,
    ExtractionResult,
    LLMExtractionResponse,
    OpinionMetadata,
    RejectedTagSuggestion,
)
from pac_pipeline.models import ParsedOpinion

log = structlog.get_logger()

_MAX_RETRIES = 2
_MAX_FULL_TEXT_CHARS = 15_000

_SYSTEM_PROMPT = (
    """\
You are extracting structured metadata from Indiana Public Access Counselor (PAC) \
advisory opinions. Given the text of an opinion, call the extract_metadata tool with \
the following fields.

Field definitions:

- issue_date: The date the opinion was issued, in YYYY-MM-DD format. Found near the \
top of the opinion after the header block.
- complainant: The name of the person or entity who filed the complaint. If redacted \
(e.g., "Letter-Redacted") or not identifiable, use null.
- public_agency: The name of the government agency that is the respondent.
- agency_type: Classify the respondent agency. Values:
  - "county": County government or county office (e.g., County Commissioners, County Clerk)
  - "city": City government or city department
  - "town": Town government
  - "school_corporation": School board, school corporation, or school district
  - "state_agency": State-level agency, board, or commission
  - "law_enforcement": Police department, sheriff, or law enforcement agency
  - "court": Court or court clerk
  - "library": Public library or library board
  - "utility": Municipal utility or utility board
  - "other": Any agency that does not fit the above categories
- statutes_invoked: List every Indiana Code section cited in the opinion (e.g., \
"IC 5-14-3-4(b)(8)"). Include subsections as written.
- statute_areas: Which areas of law are at issue. Values:
  - "APRA": Access to Public Records Act (IC 5-14-3)
  - "ODL": Open Door Law (IC 5-14-1.5)
  Include both if both are discussed substantively.
- issue_tags: Select ALL applicable tags from this controlled list. Choose every tag \
that is substantively discussed in the opinion, not merely mentioned in passing:
  """
    + ", ".join(ISSUE_TAGS)
    + """
- tags_need_review: Set to true if:
  - None of the tags from the vocabulary clearly apply to the opinion's subject matter
  - The tag assignment is approximate or uncertain
  - The opinion covers a topic not well-represented by the existing tags
  Set to false when you are confident the selected tags accurately describe the opinion.
- finding: The counselor's conclusion. Values:
  - "violation": The agency violated the law (one or more allegations sustained)
  - "no_violation": The agency did not violate the law (all allegations rejected)
  - "mixed": Some allegations were sustained AND some were rejected (requires multiple \
distinct allegations with different outcomes)
  - "moot": The issue was resolved before the opinion was issued
  - "no_jurisdiction": The PAC determined it lacked authority over the matter
  - "withdrawn": The complaint was withdrawn
- summary: 2-3 sentences summarizing the complaint and the counselor's finding. Be \
specific about the legal issues and statutes involved. Do not include advisory disclaimers.

Important rules:
- Do NOT invent information not present in the text.
- If the complainant name is redacted or unclear, set complainant to null.
- For finding, use "mixed" ONLY when there are multiple distinct allegations with \
different outcomes. A single allegation that was partially sustained should be \
"violation" or "no_violation" depending on the primary thrust.
- For informal opinions (INF), the finding is typically "no_violation" unless the \
opinion explicitly identifies a violation.
"""
)

_TOOL_DEF = {
    "name": "extract_metadata",
    "description": "Extract structured metadata from a PAC advisory opinion.",
    "input_schema": LLMExtractionResponse.model_json_schema(),
}


def build_text_payload(parsed: ParsedOpinion) -> str:
    if not parsed.sections:
        return parsed.full_text[:_MAX_FULL_TEXT_CHARS]

    first_section_start = min(start for start, _end in parsed.sections.values())
    caption = parsed.full_text[:first_section_start]

    section_order = ["BACKGROUND", "ANALYSIS", "DISCUSSION", "OPINION", "CONCLUSION"]
    section_texts: list[str] = []
    for name in section_order:
        if name in parsed.sections:
            start, end = parsed.sections[name]
            section_texts.append(f"[{name}]\n{parsed.full_text[start:end]}")

    return caption + "\n\n---\n\n" + "\n\n".join(section_texts)


def extract_one(
    opinion_id: str,
    parsed: ParsedOpinion,
    counselor_name: str | None,
    client: Anthropic,
) -> ExtractionResult:
    text_payload = build_text_payload(parsed)

    messages: list[dict[str, object]] = [
        {"role": "user", "content": text_payload},
    ]

    last_error: Exception | None = None
    rejected_tags: list[RejectedTagSuggestion] = []

    for attempt in range(_MAX_RETRIES + 1):
        response = client.messages.create(
            model=settings.extraction_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_DEF],  # type: ignore[list-item]
            tool_choice={"type": "tool", "name": "extract_metadata"},
            messages=messages,  # type: ignore[arg-type]
        )

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_block is None:
            last_error = ValueError("No tool_use block in response")
            continue

        try:
            llm_response = LLMExtractionResponse.model_validate(tool_block.input)

            invalid_tags = [t for t in llm_response.issue_tags if t not in ISSUE_TAGS_SET]
            if invalid_tags:
                all_tags = tuple(llm_response.issue_tags)
                for tag in invalid_tags:
                    rejected_tags.append(
                        RejectedTagSuggestion(
                            opinion_id=opinion_id,
                            suggested_tag=tag,
                            attempt=attempt,
                            all_suggested_tags=all_tags,
                        )
                    )

            issue_date: date | None = None
            if llm_response.issue_date:
                try:
                    issue_date = date.fromisoformat(llm_response.issue_date)
                except ValueError:
                    log.warning(
                        "invalid_issue_date",
                        opinion_id=opinion_id,
                        raw_date=llm_response.issue_date,
                    )

            metadata = OpinionMetadata(
                opinion_id=opinion_id,
                issue_date=issue_date,
                counselor=counselor_name,
                complainant=llm_response.complainant,
                public_agency=llm_response.public_agency,
                agency_type=llm_response.agency_type,
                statutes_invoked=llm_response.statutes_invoked,
                statute_areas=llm_response.statute_areas,
                issue_tags=llm_response.issue_tags,
                tags_need_review=llm_response.tags_need_review,
                finding=llm_response.finding,
                summary=llm_response.summary,
            )
            return ExtractionResult(metadata=metadata, rejected_tags=tuple(rejected_tags))
        except Exception as e:
            last_error = e
            log.warning(
                "extraction_validation_failed",
                opinion_id=opinion_id,
                attempt=attempt + 1,
                error=str(e),
            )
            messages.append({"role": "assistant", "content": response.content})  # type: ignore[dict-item]
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "is_error": True,
                            "content": f"Validation error: {e}. Please fix and try again.",
                        }
                    ],
                }
            )
            continue

    raise ValueError(
        f"Metadata extraction failed for {opinion_id} after {_MAX_RETRIES + 1} attempts: "
        f"{last_error}"
    )


def extract_all(
    db_path: Path,
    data_dir: Path,
    *,
    opinion_id: str | None = None,
    reextract: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    client = Anthropic(api_key=settings.anthropic_api_key)

    with ExtractDB(db_path) as db:
        opinions = db.get_extractable_opinions(
            opinion_id=opinion_id,
            reextract=reextract,
            limit=limit,
            extractor_version=EXTRACTOR_VERSION,
        )

        if not opinions:
            log.info("no_opinions_to_extract")
            return 0, 0

        log.info("extracting_opinions", count=len(opinions))

        extracted_count = 0
        error_count = 0

        for oid, _year, parse_path_str in tqdm(opinions, desc="Extracting"):
            try:
                parse_path = Path(parse_path_str)
                parsed = ParsedOpinion.model_validate_json(parse_path.read_text())
                counselor_name = extract_counselor(parsed.full_text)

                result = extract_one(oid, parsed, counselor_name, client)

                db.upsert_metadata(result.metadata, extractor_version=EXTRACTOR_VERSION)
                if result.rejected_tags:
                    db.insert_rejected_tags(
                        result.rejected_tags,
                        extractor_version=EXTRACTOR_VERSION,
                    )
                db.update_extract_status(oid, "ok")
                extracted_count += 1
            except Exception:
                log.exception("extraction_failed", opinion_id=oid)
                db.update_extract_status(oid, "error")
                error_count += 1

        return extracted_count, error_count
