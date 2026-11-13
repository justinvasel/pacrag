from __future__ import annotations

import json

import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel

log = structlog.get_logger()

_JUDGE_SYSTEM = """\
You are an evaluation judge for a legal question-answering system about Indiana \
Public Access Counselor advisory opinions.

Your task: compare a generated answer against a reference ideal answer and a list \
of key facts. For each key fact, determine if the generated answer substantively \
covers it (even if phrased differently).

Respond with JSON only. Do not include any text outside the JSON object.\
"""

_JUDGE_USER = """\
## Question
{question}

## Ideal Answer (reference)
{ideal_answer}

## Key Facts to Check
{numbered_facts}

## Generated Answer (to evaluate)
{generated_answer}

## Instructions
For each numbered key fact, determine if the generated answer covers it.
A fact is "covered" if the answer conveys the same information, even using \
different words. Be generous with paraphrasing, but the substance must \
be present.

Return this JSON:
{{
  "facts": [
    {{"fact_index": 1, "covered": true, "reasoning": "brief explanation"}},
    {{"fact_index": 2, "covered": false, "reasoning": "brief explanation"}}
  ]
}}\
"""


class FactJudgment(BaseModel, frozen=True):
    fact_index: int
    covered: bool
    reasoning: str


class JudgeResponse(BaseModel, frozen=True):
    facts: list[FactJudgment]


def _format_numbered_facts(key_facts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(key_facts))


def _parse_judge_response(text: str, expected_count: int) -> JudgeResponse:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    parsed = json.loads(cleaned)
    response = JudgeResponse.model_validate(parsed)

    if len(response.facts) != expected_count:
        raise ValueError(f"expected {expected_count} fact judgments, got {len(response.facts)}")

    return response


async def judge_fact_coverage(
    question: str,
    ideal_answer: str,
    key_facts: list[str],
    generated_answer: str,
    client: AsyncAnthropic,
    model: str = "claude-sonnet-4-6",
) -> tuple[float, list[str], list[str]]:
    user_msg = _JUDGE_USER.format(
        question=question,
        ideal_answer=ideal_answer,
        numbered_facts=_format_numbered_facts(key_facts),
        generated_answer=generated_answer,
    )

    for attempt in range(2):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )

            text = response.content[0].text  # type: ignore[union-attr]
            parsed = _parse_judge_response(text, len(key_facts))

            found = [key_facts[f.fact_index - 1] for f in parsed.facts if f.covered]
            missing = [key_facts[f.fact_index - 1] for f in parsed.facts if not f.covered]
            coverage = len(found) / len(key_facts) if key_facts else 1.0

            return coverage, found, missing

        except Exception:
            if attempt == 0:
                log.warning("judge_parse_failed_retrying", attempt=attempt)
                continue
            log.error("judge_failed_after_retry")
            return 0.0, [], list(key_facts)

    return 0.0, [], list(key_facts)
