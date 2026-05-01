from __future__ import annotations

import math

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from pac_api.db import OpinionDB
from pac_api.dependencies import get_db
from pac_api.schemas import (
    OpinionDetail,
    OpinionListResponse,
    OpinionMetadataResponse,
    OpinionSummary,
)

log = structlog.get_logger()
router = APIRouter()


@router.get("/opinions", response_model=OpinionListResponse)
async def list_opinions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    year_min: int | None = Query(default=None),
    year_max: int | None = Query(default=None),
    agency_type: str | None = Query(default=None),
    finding: str | None = Query(default=None),
    statute_areas: list[str] | None = Query(default=None),
    issue_tags: list[str] | None = Query(default=None),
    db: OpinionDB = Depends(get_db),
) -> OpinionListResponse:
    items, total = db.list_opinions(
        page=page,
        page_size=page_size,
        year_min=year_min,
        year_max=year_max,
        agency_type=agency_type,
        finding=finding,
        statute_areas=statute_areas,
        issue_tags=issue_tags,
    )

    total_pages = max(1, math.ceil(total / page_size))

    return OpinionListResponse(
        items=[OpinionSummary(**item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/opinions/{opinion_id}", response_model=OpinionDetail)
async def get_opinion(
    opinion_id: str,
    db: OpinionDB = Depends(get_db),
) -> OpinionDetail:
    row = db.get_opinion(opinion_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Opinion {opinion_id} not found")

    metadata = db.get_opinion_metadata(opinion_id)
    parsed = db.get_parsed_text(opinion_id)

    metadata_response: OpinionMetadataResponse | None = None
    if metadata is not None:
        metadata_response = OpinionMetadataResponse(
            issue_date=metadata.issue_date,
            counselor=metadata.counselor,
            complainant=metadata.complainant,
            public_agency=metadata.public_agency,
            agency_type=metadata.agency_type,
            statutes_invoked=list(metadata.statutes_invoked),
            statute_areas=list(metadata.statute_areas),
            issue_tags=list(metadata.issue_tags),
            finding=metadata.finding,
            summary=metadata.summary,
        )

    return OpinionDetail(
        opinion_id=opinion_id,
        year=row["year"],
        agency_name=row["agency_name"],
        pdf_url=row.get("pdf_url"),
        metadata=metadata_response,
        full_text=parsed["full_text"] if parsed else None,
        sections=parsed["sections"] if parsed else None,
    )
