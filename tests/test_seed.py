"""Seed script tests (PLAN.md Phase 2 acceptance: 'seed creates a demo portfolio')."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.clauses import CallScheduleRepository, RatingTriggerRepository
from app.db.repositories.instruments import InstrumentRepository, SukukStructureRepository
from app.db.repositories.portfolio import PortfolioHoldingRepository, PortfolioRepository
from app.db.seed import AS_OF, seed
from app.rules.ratings import rank


async def test_seed_creates_instruments_and_portfolios(db_session: AsyncSession) -> None:
    counts = await seed(db_session)

    assert counts["instruments"] == 3
    assert counts["portfolios"] == 2
    assert counts["holdings"] == 4

    portfolios = PortfolioRepository(db_session)
    green = await portfolios.get_by_name("Green Fixed Income Fund")
    assert green is not None

    holdings = PortfolioHoldingRepository(db_session)
    assert len(await holdings.list_holdings(green.id, as_of=AS_OF)) == 2


async def test_seed_is_idempotent(db_session: AsyncSession) -> None:
    """Re-running must update in place, not duplicate.

    Every seeded entity is asserted, not just the ones with a unique
    constraint: rating triggers and call schedules have no natural key, so
    their idempotency is guarded in code and can regress silently.
    """
    await seed(db_session)
    second = await seed(db_session)

    assert second == dict.fromkeys(second, 0), f"second run inserted rows: {second}"

    instruments = InstrumentRepository(db_session)
    structures = SukukStructureRepository(db_session)
    triggers = RatingTriggerRepository(db_session)
    calls = CallScheduleRepository(db_session)
    holdings = PortfolioHoldingRepository(db_session)

    assert await instruments.count() == 3
    assert await structures.count() == 3
    assert await triggers.count() == 2  # the Wakalah sukuk has none
    assert await calls.count() == 3
    assert await holdings.count() == 4


async def test_seed_three_times_still_produces_one_dataset(
    db_session: AsyncSession,
) -> None:
    """A stronger guard: repeated runs must converge, not merely not-crash."""
    for _ in range(3):
        await seed(db_session)

    assert await RatingTriggerRepository(db_session).count() == 2
    assert await CallScheduleRepository(db_session).count() == 3


async def test_ratings_are_ranked_across_both_agency_notations(
    db_session: AsyncSession,
) -> None:
    """The seed carries MARC (A-, BBB+) and RAM (AA3) deliberately."""
    await seed(db_session)
    instruments = InstrumentRepository(db_session)

    wakalah = await instruments.get_by_name("RM500m Wakalah Sukuk")
    assert wakalah is not None
    assert wakalah.current_rating == "AA3"
    # RAM's AA3 is MARC's AA- -- same notch, same rank.
    assert wakalah.current_rating_rank == rank("AA-")

    ijarah = await instruments.get_by_name("RM300m Green Ijarah Sukuk")
    assert ijarah is not None
    assert ijarah.current_rating_rank == rank("A-")


async def test_flagship_query_returns_a_strict_subset(db_session: AsyncSession) -> None:
    """'Holdings rated below A' must exclude the AA- name, not return everything."""
    await seed(db_session)
    instruments = InstrumentRepository(db_session)

    below_a = await instruments.list_rated_at_or_below(rank("A") + 1)
    names = {i.instrument_name for i in below_a}

    assert "RM300m Green Ijarah Sukuk" in names  # A-
    assert "RM250m Retail REIT Sukuk" in names  # BBB+
    assert "RM500m Wakalah Sukuk" not in names  # AA- -- comfortably above


async def test_seed_populates_sukuk_structures(db_session: AsyncSession) -> None:
    await seed(db_session)
    instruments = InstrumentRepository(db_session)
    structures = SukukStructureRepository(db_session)

    ijarah = await instruments.get_by_name("RM300m Green Ijarah Sukuk")
    assert ijarah is not None
    structure = await structures.get_for_instrument(ijarah.id)

    assert structure is not None
    assert structure.purchase_undertaking is True
    # Shariah non-compliance is modelled as a structured event, not free text.
    events = structure.shariah_compliance_events_json
    assert any(e.get("triggers_purchase_undertaking") for e in events)


async def test_overlapping_holding_appears_in_both_portfolios(
    db_session: AsyncSession,
) -> None:
    """Exposure aggregation must actually aggregate across funds."""
    await seed(db_session)
    portfolios = PortfolioRepository(db_session)
    holdings = PortfolioHoldingRepository(db_session)
    instruments = InstrumentRepository(db_session)

    ijarah = await instruments.get_by_name("RM300m Green Ijarah Sukuk")
    assert ijarah is not None

    total = Decimal("0")
    for name in ("Green Fixed Income Fund", "Income Growth Fund"):
        pf = await portfolios.get_by_name(name)
        assert pf is not None
        for holding in await holdings.list_holdings(pf.id, as_of=AS_OF):
            if holding.instrument_id == ijarah.id and holding.market_value is not None:
                total += holding.market_value

    assert total == Decimal("35000000.0000")
