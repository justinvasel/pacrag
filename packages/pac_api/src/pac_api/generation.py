from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from anthropic import AsyncAnthropic
from pac_pipeline.config import settings
from pac_pipeline.retrieve_models import RetrievedChunk

log = structlog.get_logger()

SYSTEM_PROMPT = """\
You are an expert on Indiana Public Access Counselor (PAC) advisory opinions, \
which interpret Indiana's Access to Public Records Act (APRA, IC 5-14-3) \
and Open Door Law (ODL, IC 5-14-1.5).

You answer questions by analyzing the retrieved PAC opinion excerpts provided below.

RULES:
1. Every factual claim must be cited in the form (opinion_id, p. N) where opinion_id \
is the identifier and N is the page number from the retrieved excerpt.
2. Use ONLY information from the provided excerpts. Do not fabricate citations or \
reference opinions not in the excerpts.
3. If the retrieved opinions don't directly address the question, say: \
"The retrieved opinions don't directly address this."
4. End every answer with this disclaimer on its own line: \
"Note: PAC opinions are advisory and persuasive only, not binding on agencies or courts."
"""

_MAX_TOKENS = 2048


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    parts = []
    for rc in chunks:
        c = rc.chunk
        header = f"[Opinion {c.opinion_id}, p. {c.page_num}, {c.section}]"
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


async def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
    client: AsyncAnthropic,
) -> AsyncGenerator[tuple[str, str], None]:
    context = format_context(chunks)
    user_message = f"Retrieved excerpts:\n\n{context}\n\nQuestion: {question}"

    full_text_parts: list[str] = []

    async with client.messages.stream(
        model=settings.generation_model,
        max_tokens=_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            full_text_parts.append(text)
            yield ("token", text)

    yield ("done", "".join(full_text_parts))
