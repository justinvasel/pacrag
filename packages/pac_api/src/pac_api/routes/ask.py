from __future__ import annotations

import json

import structlog
from anthropic import AsyncAnthropic
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pac_pipeline import retrieve as do_retrieve

from pac_api.citations import check_response
from pac_api.dependencies import get_anthropic
from pac_api.generation import stream_answer
from pac_api.schemas import AskRequest, AskResponseMeta, chunk_to_response

log = structlog.get_logger()
router = APIRouter()


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/ask")
async def ask(
    request: AskRequest,
    client: AsyncAnthropic = Depends(get_anthropic),
) -> StreamingResponse:
    result = await do_retrieve(request.question, filters=request.filters)

    if not result.chunks:
        no_result_meta = AskResponseMeta(
            citations=[],
            applied_filters=result.applied_filters,
            relaxed=result.relaxed,
            chunks=[],
        )

        async def empty_stream():  # type: ignore[return]
            yield _sse_event(
                "token",
                json.dumps({"token": "No relevant opinions were found for this question."}),
            )
            yield _sse_event("metadata", no_result_meta.model_dump_json())

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    retrieved_ids = frozenset(rc.chunk.opinion_id for rc in result.chunks)
    chunk_responses = [chunk_to_response(rc) for rc in result.chunks]

    async def event_stream():  # type: ignore[return]
        full_text = ""
        try:
            async for kind, content in stream_answer(request.question, result.chunks, client):
                if kind == "token":
                    yield _sse_event("token", json.dumps({"token": content}))
                elif kind == "done":
                    full_text = content
        except Exception:
            log.exception("stream_answer_failed", question=request.question)
            yield _sse_event("error", json.dumps({"error": "Generation failed"}))
            return

        citations, all_valid = check_response(full_text, retrieved_ids)

        meta = AskResponseMeta(
            citations=citations,
            applied_filters=result.applied_filters,
            relaxed=result.relaxed,
            chunks=chunk_responses,
        )
        yield _sse_event("metadata", meta.model_dump_json())

        if not all_valid:
            invalid = [c.model_dump() for c in citations if not c.valid]
            yield _sse_event(
                "error",
                json.dumps({"error": "Citation validation failed", "invalid_citations": invalid}),
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
