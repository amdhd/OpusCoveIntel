"""Read schemas for the covenant catalogue — instruments, covenants, provenance.

Separate from the ORM models for the same reason `documents.py` is: the wire
format is a deliberate choice, not whatever the table happens to hold.

One rule shapes everything here. **A covenant that cannot name its source page
and verbatim quote is invalid** (CLAUDE.md 1.2), so `CovenantRead` carries a
non-optional `source` rather than an optional `clause_id` for the caller to
chase. A reader cannot accidentally render a threshold without its evidence,
because the evidence is not a separate fetch.

`domain/` is a pure leaf -- no db, no llm, no I/O (CLAUDE.md 3).
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import (
    CallType,
    ClauseType,
    CovenantType,
    ExtractionMethod,
    InstrumentType,
    RatingAgency,
    ReviewStatus,
    Severity,
    SukukStructureType,
    TriggerDirection,
)


class ClauseSource(BaseModel):
    """Where a structured fact came from, in enough detail to go look.

    `char_start` / `char_end` are what let a viewer highlight the exact span
    inside the chunk rather than merely naming the page.
    """

    model_config = ConfigDict(from_attributes=True)

    clause_id: uuid.UUID
    document_id: uuid.UUID
    chunk_id: uuid.UUID | None = None
    clause_type: ClauseType
    page_number: int = Field(ge=1)
    section_title: str | None = None
    source_quote: str
    char_start: int | None = None
    char_end: int | None = None
    citation_verified: bool
    citation_match_score: float | None = None


class CovenantRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID | None = None
    covenant_type: CovenantType
    summary: str | None = None
    conditions: dict[str, object] = Field(default_factory=dict)
    thresholds: dict[str, object] = Field(default_factory=dict)
    threshold_amount: Decimal | None = None
    threshold_currency: str | None = None
    effective_date: dt.date | None = None
    trigger_event: str | None = None
    severity: Severity
    method: ExtractionMethod
    confidence: float | None = None
    review_status: ReviewStatus
    source: ClauseSource


class CallScheduleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    call_date: dt.date
    call_price: Decimal | None = None
    call_type: CallType
    method: ExtractionMethod
    confidence: float | None = None
    review_status: ReviewStatus
    source_clause_id: uuid.UUID | None = None


class RatingTriggerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    rating_agency: RatingAgency
    trigger_rating: str
    # The ordinal notch rank. Exposed because rating comparison is ordinal, not
    # lexical (CLAUDE.md 6) -- a client sorting on `trigger_rating` would put
    # 'AA-' below 'A+'. Sort on this instead.
    trigger_rank: int
    trigger_direction: TriggerDirection
    consequence: str
    severity: Severity
    method: ExtractionMethod
    confidence: float | None = None
    review_status: ReviewStatus
    source_clause_id: uuid.UUID | None = None


class SukukStructureRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    structure_type: SukukStructureType
    spv_name: str | None = None
    originator: str | None = None
    underlying_asset: str | None = None
    profit_rate: Decimal | None = None
    profit_payment_frequency: str | None = None
    # Shariah non-compliance is a dissolution event that triggers a purchase
    # undertaking (CLAUDE.md 6). These stay distinct fields, not one blob.
    purchase_undertaking: bool | None = None
    dissolution_events: list[dict[str, object]] = Field(default_factory=list)
    shariah_compliance_events: list[dict[str, object]] = Field(default_factory=list)
    confidence: float | None = None
    review_status: ReviewStatus


class InstrumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    issuer_name: str
    instrument_name: str
    instrument_type: InstrumentType
    currency: str
    isin: str | None = None
    ticker: str | None = None
    sukuk_structure: SukukStructureType
    issue_size: Decimal | None = None
    maturity_date: dt.date | None = None
    current_rating: str | None = None
    current_rating_rank: int | None = None
    rating_agency: RatingAgency
    review_status: ReviewStatus


class InstrumentDetail(InstrumentRead):
    """An instrument with everything that hangs off it.

    Assembled in one response because the screens that need an instrument need
    its covenants, calls and triggers together; four round trips to render one
    page is a worse contract than one slightly larger payload.
    """

    sukuk_structure_detail: SukukStructureRead | None = None
    covenants: list[CovenantRead] = Field(default_factory=list)
    call_schedules: list[CallScheduleRead] = Field(default_factory=list)
    rating_triggers: list[RatingTriggerRead] = Field(default_factory=list)


class ClauseProvenance(BaseModel):
    """A clause plus the chunk text it was cut from.

    This is what makes the citation chain inspectable rather than merely
    recorded: `chunk_text` is the surrounding context, and
    `quote_start` / `quote_end` locate the quote *within that text* so a viewer
    can highlight it without re-deriving offsets.

    The offsets on `Clause` are relative to the document, not to the chunk, so
    they cannot be used for that directly -- these are resolved by the service.
    """

    model_config = ConfigDict(from_attributes=True)

    source: ClauseSource
    clause_text: str
    method: ExtractionMethod
    confidence: float | None = None
    review_status: ReviewStatus
    document_filename: str | None = None
    chunk_text: str | None = None
    quote_start: int | None = None
    quote_end: int | None = None
    covenants: list[CovenantRead] = Field(default_factory=list)


class HoldingRead(BaseModel):
    """A position, with the instrument resolved.

    `market_value` and `nav_weight` stay `Decimal` to the wire, where Pydantic
    renders them as JSON **strings**. That is what keeps CLAUDE.md 6 true past
    the boundary: a numeric literal would be parsed into a double by any
    browser, undoing the exactness Postgres and Python maintained. Clients must
    parse these with a decimal type, not `Number()`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument: InstrumentRead
    quantity: Decimal | None = None
    market_value: Decimal | None = None
    nav_weight: Decimal | None = None
    as_of_date: dt.date
    source: str | None = None


class PortfolioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    owner: str | None = None
    mandate_type: str | None = None
    base_currency: str


class PortfolioHoldings(BaseModel):
    """Holdings for one portfolio on one date.

    `as_of_date` is echoed because holdings are a time series and the default
    is "latest available" -- a caller that does not know which date it got
    cannot report the number responsibly.
    """

    portfolio: PortfolioRead
    as_of_date: dt.date | None = None
    holdings: list[HoldingRead] = Field(default_factory=list)
    total_market_value: Decimal | None = None
    count: int
