"""Synthetic demo data.

CLAUDE.md 7: synthetic documents only -- real prospectuses are copyrighted.
Every issuer below is invented; any resemblance to a real Malaysian issuer is
coincidental and unintended.

The dataset is shaped to exercise the queries that matter:

* Three sukuk across three Shariah structures (Ijarah, Wakalah, Musharakah).
* Ratings in **both** agency notations -- MARC's `A-` / `BBB+` and RAM's `AA3`
  -- so the ordinal rank normaliser is exercised, not just assumed.
* Rating triggers at different notches, so "downgraded below A" returns a
  strict subset rather than everything.
* Two portfolios with an overlapping holding, so exposure aggregation has to
  actually aggregate.

Idempotent: re-running updates in place rather than duplicating.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.clauses import CallSchedule, RatingTrigger
from app.db.models.instruments import Instrument, SukukStructure
from app.db.models.portfolio import Portfolio, PortfolioHolding
from app.db.repositories.clauses import CallScheduleRepository, RatingTriggerRepository
from app.db.repositories.instruments import InstrumentRepository, SukukStructureRepository
from app.db.repositories.portfolio import PortfolioHoldingRepository, PortfolioRepository
from app.db.session import dispose_engines, get_sessionmaker
from app.domain.enums import (
    CallType,
    ExtractionMethod,
    InstrumentType,
    RatingAgency,
    Severity,
    SukukStructureType,
    TriggerDirection,
)
from app.rules.ratings import rank

logger = get_logger(__name__)

AS_OF = dt.date(2026, 1, 1)

INSTRUMENTS: list[dict[str, Any]] = [
    {
        "issuer_name": "Synthetic Green Energy Sdn Bhd",
        "instrument_name": "RM300m Green Ijarah Sukuk",
        "instrument_type": InstrumentType.SUKUK,
        "sukuk_structure": SukukStructureType.IJARAH,
        "issue_size": Decimal("300000000"),
        "maturity_date": dt.date(2030, 6, 15),
        "rating_agency": RatingAgency.MARC,
        # MARC notation.
        "current_rating": "A-",
        "isin": "MYSYN0000001",
        "structure": {
            "spv_name": "Green Ijarah Capital Bhd",
            "originator": "Synthetic Green Energy Sdn Bhd",
            "underlying_asset": "Portfolio of operating solar assets in Kedah and Perlis",
            "profit_rate": Decimal("4.850000"),
            "profit_payment_frequency": "semi-annual",
            "purchase_undertaking": True,
            "dissolution_events_json": [
                {"event": "shariah_non_compliance", "consequence": "dissolution"},
                {"event": "payment_default", "grace_period_days": 7},
            ],
            "shariah_compliance_events_json": [
                {
                    "event": "shariah_non_compliance",
                    "triggers_purchase_undertaking": True,
                }
            ],
        },
        # Investors may require early redemption if the issuer falls below A.
        "rating_trigger": {"trigger_rating": "A", "consequence": "Investor put at par"},
        "call": {"call_date": dt.date(2027, 6, 15), "call_price": Decimal("100.000000")},
    },
    {
        "issuer_name": "Synthetic Infrastructure Holdings Berhad",
        "instrument_name": "RM500m Wakalah Sukuk",
        "instrument_type": InstrumentType.SUKUK,
        "sukuk_structure": SukukStructureType.WAKALAH,
        "issue_size": Decimal("500000000"),
        "maturity_date": dt.date(2031, 12, 1),
        "rating_agency": RatingAgency.RAM,
        # RAM notation -- normalises to "AA-", not a literal table lookup.
        "current_rating": "AA3",
        "isin": "MYSYN0000002",
        "structure": {
            "spv_name": "Infra Wakalah Bhd",
            "originator": "Synthetic Infrastructure Holdings Berhad",
            "underlying_asset": "Concession receivables from a tolled expressway",
            "profit_rate": Decimal("4.200000"),
            "profit_payment_frequency": "semi-annual",
            "purchase_undertaking": True,
            "dissolution_events_json": [{"event": "payment_default", "grace_period_days": 5}],
            "shariah_compliance_events_json": [],
        },
        "rating_trigger": None,  # "Not applicable" in the source spec.
        "call": {"call_date": dt.date(2026, 12, 1), "call_price": Decimal("101.500000")},
    },
    {
        "issuer_name": "Synthetic Retail REIT Berhad",
        "instrument_name": "RM250m Retail REIT Sukuk",
        "instrument_type": InstrumentType.SUKUK,
        "sukuk_structure": SukukStructureType.MUSHARAKAH,
        "issue_size": Decimal("250000000"),
        "maturity_date": dt.date(2032, 3, 31),
        "rating_agency": RatingAgency.MARC,
        "current_rating": "BBB+",
        "isin": "MYSYN0000003",
        "structure": {
            "spv_name": "Retail Musharakah Bhd",
            "originator": "Synthetic Retail REIT Berhad",
            "underlying_asset": "Beneficial interest in two suburban retail malls",
            "profit_rate": Decimal("5.400000"),
            "profit_payment_frequency": "quarterly",
            "purchase_undertaking": True,
            "dissolution_events_json": [
                {"event": "shariah_non_compliance", "consequence": "dissolution"}
            ],
            "shariah_compliance_events_json": [
                {"event": "shariah_non_compliance", "triggers_purchase_undertaking": True}
            ],
        },
        "rating_trigger": {
            "trigger_rating": "BBB",
            "consequence": "Mandatory early purchase undertaking",
        },
        "call": {"call_date": dt.date(2028, 3, 31), "call_price": Decimal("100.000000")},
    },
]

PORTFOLIOS: list[dict[str, Any]] = [
    {
        "name": "Green Fixed Income Fund",
        "owner": "Fixed Income Desk",
        "mandate_type": "shariah_compliant",
        "holdings": [
            ("RM300m Green Ijarah Sukuk", Decimal("25000000"), Decimal("0.120000")),
            ("RM500m Wakalah Sukuk", Decimal("18000000"), Decimal("0.090000")),
        ],
    },
    {
        "name": "Income Growth Fund",
        "owner": "Fixed Income Desk",
        "mandate_type": "balanced",
        "holdings": [
            ("RM250m Retail REIT Sukuk", Decimal("30000000"), Decimal("0.150000")),
            # Deliberate overlap with the fund above.
            ("RM300m Green Ijarah Sukuk", Decimal("10000000"), Decimal("0.050000")),
        ],
    },
]


async def seed(session: AsyncSession) -> dict[str, int]:
    """Populate synthetic data. Safe to run repeatedly."""
    instruments = InstrumentRepository(session)
    structures = SukukStructureRepository(session)
    triggers = RatingTriggerRepository(session)
    calls = CallScheduleRepository(session)
    portfolios = PortfolioRepository(session)
    holdings = PortfolioHoldingRepository(session)

    counts = {"instruments": 0, "portfolios": 0, "holdings": 0, "rating_triggers": 0, "calls": 0}
    by_name: dict[str, Instrument] = {}

    for spec in INSTRUMENTS:
        existing = await instruments.get_by_name(spec["instrument_name"])
        if existing is None:
            existing = Instrument(
                issuer_name=spec["issuer_name"],
                instrument_name=spec["instrument_name"],
                instrument_type=spec["instrument_type"],
                sukuk_structure=spec["sukuk_structure"],
                issue_size=spec["issue_size"],
                maturity_date=spec["maturity_date"],
                rating_agency=spec["rating_agency"],
                isin=spec["isin"],
                currency="MYR",
            )
            await instruments.add(existing)
            counts["instruments"] += 1

        # Always route ratings through the repository so the ordinal rank stays
        # consistent with the rating string.
        await instruments.set_rating(existing, spec["current_rating"])
        by_name[spec["instrument_name"]] = existing

        # Look up by instrument_id, not by primary key -- SukukStructure has its
        # own id, so session.get(SukukStructure, instrument.id) would always miss.
        if await structures.get_for_instrument(existing.id) is None:
            session.add(
                SukukStructure(
                    instrument_id=existing.id,
                    structure_type=spec["sukuk_structure"],
                    confidence=1.0,
                    **spec["structure"],
                )
            )

        # Rating triggers and call schedules have no natural unique key (an
        # instrument can legitimately carry several of each), so idempotency
        # cannot be delegated to a constraint -- guard on "already has any".
        trigger_spec = spec["rating_trigger"]
        if trigger_spec is not None and await triggers.count(instrument_id=existing.id) == 0:
            session.add(
                RatingTrigger(
                    instrument_id=existing.id,
                    rating_agency=spec["rating_agency"],
                    trigger_rating=trigger_spec["trigger_rating"],
                    # Rank is derived, never hand-written.
                    trigger_rank=rank(trigger_spec["trigger_rating"], spec["rating_agency"]),
                    trigger_direction=TriggerDirection.DOWNGRADE_BELOW,
                    consequence=trigger_spec["consequence"],
                    severity=Severity.HIGH,
                    method=ExtractionMethod.HUMAN,
                    confidence=1.0,
                )
            )
            counts["rating_triggers"] += 1

        call = spec["call"]
        if await calls.count(instrument_id=existing.id) == 0:
            session.add(
                CallSchedule(
                    instrument_id=existing.id,
                    call_date=call["call_date"],
                    call_price=call["call_price"],
                    call_type=CallType.OPTIONAL,
                    method=ExtractionMethod.HUMAN,
                    confidence=1.0,
                )
            )
            counts["calls"] += 1

    await session.flush()

    for spec in PORTFOLIOS:
        portfolio = await portfolios.get_by_name(spec["name"])
        if portfolio is None:
            portfolio = Portfolio(
                name=spec["name"],
                owner=spec["owner"],
                mandate_type=spec["mandate_type"],
                base_currency="MYR",
            )
            await portfolios.add(portfolio)
            counts["portfolios"] += 1

        existing_holdings = {
            h.instrument_id for h in await holdings.list_holdings(portfolio.id, as_of=AS_OF)
        }
        for instrument_name, market_value, weight in spec["holdings"]:
            instrument = by_name[instrument_name]
            if instrument.id in existing_holdings:
                continue
            session.add(
                PortfolioHolding(
                    portfolio_id=portfolio.id,
                    instrument_id=instrument.id,
                    market_value=market_value,
                    nav_weight=weight,
                    as_of_date=AS_OF,
                    source="synthetic_seed",
                )
            )
            counts["holdings"] += 1

    await session.flush()
    return counts


async def run_seed() -> dict[str, int]:
    async with get_sessionmaker()() as session:
        counts = await seed(session)
        await session.commit()
    logger.info("seed complete", extra=counts)
    return counts


async def _main() -> dict[str, int]:
    try:
        return await run_seed()
    finally:
        await dispose_engines()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
