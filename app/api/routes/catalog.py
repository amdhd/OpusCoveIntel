"""Read endpoints for instruments, covenants, provenance and portfolios.

Everything a reader needs to go from "the agent said X" to "here is the page it
read". Handlers translate HTTP and nothing else (CLAUDE.md 3, 9); `CatalogService`
decides what a covenant-with-evidence is.

These are reads, so they run on the **read-only role** -- same grants the query
agent holds. A read endpoint has no business holding write privileges, and
routing them through `get_readonly_session` means a bug that tried to write
fails at the database rather than at review.

Endpoints:
    GET /instruments                     — list
    GET /instruments/{id}                — detail, with covenants/calls/triggers
    GET /instruments/{id}/covenants      — covenants, optionally by type
    GET /clauses/{id}                    — provenance: quote, chunk, offsets
    GET /portfolios                      — list
    GET /portfolios/{id}/holdings        — positions with exposure
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.service import CatalogService
from app.db.session import get_readonly_session
from app.domain.catalog import (
    ClauseProvenance,
    CovenantRead,
    InstrumentDetail,
    InstrumentRead,
    PortfolioHoldings,
    PortfolioRead,
)
from app.domain.enums import CovenantType

router = APIRouter(tags=["catalog"])

MAX_PAGE_SIZE = 200


def get_catalog_service(
    session: Annotated[AsyncSession, Depends(get_readonly_session)],
) -> CatalogService:
    return CatalogService(session)


ServiceDep = Annotated[CatalogService, Depends(get_catalog_service)]


# -- instruments -------------------------------------------------------------


@router.get("/instruments", response_model=list[InstrumentRead], summary="List instruments")
async def list_instruments(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[InstrumentRead]:
    return await service.list_instruments(limit=limit, offset=offset)


@router.get(
    "/instruments/{instrument_id}",
    response_model=InstrumentDetail,
    summary="One instrument with its covenants, calls and rating triggers",
)
async def get_instrument(instrument_id: uuid.UUID, service: ServiceDep) -> InstrumentDetail:
    detail = await service.get_instrument(instrument_id)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="instrument not found")
    return detail


@router.get(
    "/instruments/{instrument_id}/covenants",
    response_model=list[CovenantRead],
    summary="Covenants for an instrument, each with its source clause",
)
async def list_instrument_covenants(
    instrument_id: uuid.UUID,
    service: ServiceDep,
    covenant_type: CovenantType | None = None,
) -> list[CovenantRead]:
    if await service.instruments.get(instrument_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="instrument not found")
    return await service.covenants_for_instrument(instrument_id, covenant_type=covenant_type)


@router.get(
    "/documents/{document_id}/covenants",
    response_model=list[CovenantRead],
    summary="Covenants extracted from a document",
)
async def list_document_covenants(
    document_id: uuid.UUID, service: ServiceDep
) -> list[CovenantRead]:
    if await service.documents.get(document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="document not found")
    return await service.covenants_for_document(document_id)


# -- provenance --------------------------------------------------------------


@router.get(
    "/clauses/{clause_id}",
    response_model=ClauseProvenance,
    summary="A clause with its chunk text and the quote's offsets within it",
)
async def get_clause(clause_id: uuid.UUID, service: ServiceDep) -> ClauseProvenance:
    """The provenance endpoint — what turns a citation into something checkable.

    `quote_start`/`quote_end` locate the quote inside `chunk_text`. They are
    null when the quote could not be located, and a client must then render the
    quote without highlighting rather than falling back to an approximate span.
    """
    provenance = await service.clause_provenance(clause_id)
    if provenance is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="clause not found")
    return provenance


# -- portfolios --------------------------------------------------------------


@router.get("/portfolios", response_model=list[PortfolioRead], summary="List portfolios")
async def list_portfolios(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
) -> list[PortfolioRead]:
    return await service.list_portfolios(limit=limit)


@router.get(
    "/portfolios/{portfolio_id}/holdings",
    response_model=PortfolioHoldings,
    summary="Positions and exposure for a portfolio",
)
async def get_portfolio_holdings(
    portfolio_id: uuid.UUID,
    service: ServiceDep,
    as_of: dt.date | None = None,
) -> PortfolioHoldings:
    """Holdings for a date, defaulting to the most recent available.

    The response echoes `as_of_date`, because holdings are a time series and a
    caller that does not know which date it received cannot report the exposure
    responsibly.
    """
    holdings = await service.portfolio_holdings(portfolio_id, as_of=as_of)
    if holdings is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="portfolio not found")
    return holdings
