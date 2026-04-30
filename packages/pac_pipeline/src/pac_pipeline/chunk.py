from __future__ import annotations

from typing import Any, Protocol

from pac_pipeline.chunk_models import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    Chunk,
)
from pac_pipeline.extract_models import OpinionMetadata
from pac_pipeline.models import ParsedOpinion, ParsedPage


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


_SECTION_ORDER = ["BACKGROUND", "ANALYSIS", "DISCUSSION", "OPINION", "CONCLUSION"]
_ANALYSIS_SECTIONS = {"ANALYSIS", "DISCUSSION", "OPINION"}


def count_tokens(text: str, tokenizer: Any) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def _truncate_to_limit(text: str, tokenizer: Any, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def split_paragraphs_into_chunks(
    paragraphs: list[str],
    tokenizer: Any,
    *,
    target_min: int = MIN_CHUNK_TOKENS,
    target_max: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, tokenizer)

        if current_tokens + para_tokens > target_max and current_paras:
            chunks.append("\n\n".join(current_paras))
            current_paras, current_tokens = _compute_overlap(
                current_paras, tokenizer, overlap_tokens
            )

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


def _compute_overlap(
    paras: list[str],
    tokenizer: Any,
    overlap_tokens: int,
) -> tuple[list[str], int]:
    overlap_paras: list[str] = []
    overlap_count = 0
    for para in reversed(paras):
        para_tokens = count_tokens(para, tokenizer)
        if overlap_count + para_tokens > overlap_tokens and overlap_paras:
            break
        overlap_paras.insert(0, para)
        overlap_count += para_tokens
    return overlap_paras, overlap_count


def _find_page_num(char_offset: int, pages: list[ParsedPage]) -> int:
    cumulative = 0
    for page in pages:
        if cumulative + page.char_count > char_offset:
            return page.page_num
        cumulative += page.char_count
    return pages[-1].page_num if pages else 1


def _build_card_text(metadata: OpinionMetadata) -> str:
    parts = [
        f"Opinion {metadata.opinion_id}",
        f"Agency: {metadata.public_agency} ({metadata.agency_type})",
    ]
    if metadata.complainant:
        parts.append(f"Complainant: {metadata.complainant}")
    if metadata.counselor:
        parts.append(f"Counselor: {metadata.counselor}")
    if metadata.issue_date:
        parts.append(f"Date: {metadata.issue_date.isoformat()}")
    parts.append(f"Finding: {metadata.finding}")
    if metadata.statute_areas:
        parts.append(f"Statute areas: {', '.join(metadata.statute_areas)}")
    if metadata.issue_tags:
        parts.append(f"Issue tags: {', '.join(metadata.issue_tags)}")
    parts.append("")
    parts.append(metadata.summary)
    return "\n".join(parts)


def chunk_opinion(
    parsed: ParsedOpinion,
    metadata: OpinionMetadata,
    *,
    year: int,
    tokenizer: Any,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0

    denorm = {
        "opinion_id": parsed.opinion_id,
        "year": year,
        "agency_type": metadata.agency_type,
        "statute_areas": tuple(metadata.statute_areas),
        "finding": metadata.finding,
        "issue_tags": tuple(metadata.issue_tags),
    }

    card_text = _build_card_text(metadata)
    card_text = _truncate_to_limit(card_text, tokenizer, MAX_CHUNK_TOKENS)
    chunks.append(
        Chunk(
            chunk_id=f"{parsed.opinion_id}#c{idx}",
            chunk_index=idx,
            section="card",
            page_num=1,
            text=card_text,
            token_count=count_tokens(card_text, tokenizer),
            **denorm,
        )
    )
    idx += 1

    if parsed.sections:
        for section_name in _SECTION_ORDER:
            if section_name not in parsed.sections:
                continue
            start, end = parsed.sections[section_name]
            section_text = parsed.full_text[start:end].strip()
            if not section_text:
                continue

            page_num = _find_page_num(start, parsed.pages)
            section_label = section_name.lower()

            if section_name in _ANALYSIS_SECTIONS:
                paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
                chunk_texts = split_paragraphs_into_chunks(paragraphs, tokenizer)
            else:
                token_count = count_tokens(section_text, tokenizer)
                if token_count > MAX_CHUNK_TOKENS:
                    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
                    chunk_texts = split_paragraphs_into_chunks(paragraphs, tokenizer)
                else:
                    chunk_texts = [section_text]

            for text in chunk_texts:
                chunks.append(
                    Chunk(
                        chunk_id=f"{parsed.opinion_id}#c{idx}",
                        chunk_index=idx,
                        section=section_label,
                        page_num=page_num,
                        text=text,
                        token_count=count_tokens(text, tokenizer),
                        **denorm,
                    )
                )
                idx += 1
    else:
        full_text = parsed.full_text.strip()
        if full_text:
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            chunk_texts = split_paragraphs_into_chunks(paragraphs, tokenizer)
            for text in chunk_texts:
                chunks.append(
                    Chunk(
                        chunk_id=f"{parsed.opinion_id}#c{idx}",
                        chunk_index=idx,
                        section="full_text",
                        page_num=1,
                        text=text,
                        token_count=count_tokens(text, tokenizer),
                        **denorm,
                    )
                )
                idx += 1

    return chunks
