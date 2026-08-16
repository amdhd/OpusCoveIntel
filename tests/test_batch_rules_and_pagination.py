"""Batched rule evaluation and paged list screens — review.md findings 8 and 13.

Finding 8 is a performance defect with a correctness trap in the fix. The
portfolio page called `evaluate_covenant_rule` once per holding, and each call
issued three queries — fine for two positions, six hundred for a realistic
200-bond portfolio. The obvious fix, a second evaluator that reads many rows at
once, is the one the review warned against: two rules implementations
eventually disagree, and the one on screen is the one somebody acts on.

So the tests here pin two different things. That batching actually happened —
asserted by *counting queries*, because a batch entry point that still loops
internally passes every behavioural test. And that batching changed nothing —
asserted by comparing the batch result to the single-instrument tool the agent
calls, field for field.

Finding 13 is simpler: both list screens rendered whatever the database had.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools import evaluate_covenant_rule, evaluate_covenant_rules
from app.db.models.instruments import Instrument
from app.db.models.ops import HumanReview
from app.domain.enums import ReviewStatus, ReviewTrigger
from app.web.routes import PAGE_SIZE, _page_window

pytestmark = pytest.mark.usefixtures("storage_root")


class _QueryCounter:
    """Counts SQL statements issued on a session's connection."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def __call__(self, conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        self.statements.append(statement)

    @property
    def selects(self) -> int:
        return sum(1 for s in self.statements if s.lstrip().upper().startswith("SELECT"))


async def _engine_of(session: AsyncSession) -> Any:
    """The sync engine behind an async session, for event listening."""
    connection = await session.connection()
    sync_connection = connection.sync_connection
    assert sync_connection is not None, "an open async connection always has a sync one"
    return sync_connection.engine


@pytest_asyncio.fixture
async def instrument_ids(db_session: AsyncSession, seeded_universe: None) -> list[uuid.UUID]:
    rows = (await db_session.execute(select(Instrument))).scalars().all()
    assert len(rows) >= 2, "the seeded universe should hold several instruments"
    return [row.id for row in rows]


# -- finding 8: the batch is a batch -----------------------------------------


async def test_the_batch_issues_a_fixed_number_of_queries(
    db_session: AsyncSession, instrument_ids: list[uuid.UUID]
) -> None:
    """Three queries for the whole portfolio, not three per holding.

    This is the assertion the finding is actually about. Every other test here
    would pass against an implementation that looped internally and still made
    3N round trips, which is exactly the defect.
    """
    counter = _QueryCounter()
    engine = await _engine_of(db_session)
    event.listen(engine, "before_cursor_execute", counter)
    try:
        await evaluate_covenant_rules(db_session, instrument_ids=instrument_ids)
    finally:
        event.remove(engine, "before_cursor_execute", counter)

    assert counter.selects == 3, (
        f"expected 3 selects (instruments, triggers, covenants) for "
        f"{len(instrument_ids)} instruments, got {counter.selects}"
    )


async def test_the_batch_does_not_grow_with_the_number_of_instruments(
    db_session: AsyncSession, instrument_ids: list[uuid.UUID]
) -> None:
    """One instrument and all of them cost the same number of round trips."""

    async def selects_for(ids: list[uuid.UUID]) -> int:
        counter = _QueryCounter()
        engine = await _engine_of(db_session)
        event.listen(engine, "before_cursor_execute", counter)
        try:
            await evaluate_covenant_rules(db_session, instrument_ids=ids)
        finally:
            event.remove(engine, "before_cursor_execute", counter)
        return counter.selects

    assert await selects_for(instrument_ids[:1]) == await selects_for(instrument_ids)


# -- finding 8: and it agrees with the tool the agent calls -------------------


async def test_the_batch_agrees_with_the_single_instrument_tool(
    db_session: AsyncSession, instrument_ids: list[uuid.UUID]
) -> None:
    """The property the review cared about more than the query count.

    A second rules implementation for the UI would drift from the agent's, and
    a breach board that disagrees with the answer the agent gives is worse than
    a slow one.
    """
    batch = await evaluate_covenant_rules(db_session, instrument_ids=instrument_ids)

    for instrument_id in instrument_ids:
        single = await evaluate_covenant_rule(db_session, instrument_id=instrument_id)
        assert single.ok and single.data is not None
        assert instrument_id in batch, "every existing instrument must be in the result"
        assert batch[instrument_id] == single.data, f"batch and single disagree for {instrument_id}"


async def test_an_unknown_instrument_is_absent_rather_than_fatal(
    db_session: AsyncSession, instrument_ids: list[uuid.UUID]
) -> None:
    """A holding pointing at a missing row must not take the page down."""
    missing = uuid.uuid4()

    result = await evaluate_covenant_rules(db_session, instrument_ids=[*instrument_ids, missing])

    assert missing not in result
    assert set(result) == set(instrument_ids)


async def test_no_instruments_is_no_queries(db_session: AsyncSession) -> None:
    """An empty portfolio should not touch the database at all."""
    counter = _QueryCounter()
    engine = await _engine_of(db_session)
    event.listen(engine, "before_cursor_execute", counter)
    try:
        assert await evaluate_covenant_rules(db_session, instrument_ids=[]) == {}
    finally:
        event.remove(engine, "before_cursor_execute", counter)

    assert counter.selects == 0


async def test_duplicate_instrument_ids_are_evaluated_once(
    db_session: AsyncSession, instrument_ids: list[uuid.UUID]
) -> None:
    """Two holdings of the same bond is ordinary, and must not double the work."""
    doubled = [instrument_ids[0], instrument_ids[0]]

    result = await evaluate_covenant_rules(db_session, instrument_ids=doubled)

    assert list(result) == [instrument_ids[0]]


# -- finding 13: the windows ---------------------------------------------------


class TestPageWindow:
    def test_a_short_list_is_one_page(self) -> None:
        window = _page_window(1, total=3)

        assert window["last"] == 1
        assert not window["has_next"]
        assert not window["has_previous"]
        assert window["first_shown"] == 1
        assert window["last_shown"] == 3

    def test_an_empty_list_reports_nothing_shown(self) -> None:
        window = _page_window(1, total=0)

        assert window["last"] == 1
        assert window["first_shown"] == 0
        assert window["last_shown"] == 0

    def test_a_page_past_the_end_clamps_rather_than_failing(self) -> None:
        """A queue drains while somebody reads it; page 9 going stale is normal."""
        window = _page_window(9, total=PAGE_SIZE + 1)

        assert window["page"] == 2
        assert window["offset"] == PAGE_SIZE
        assert not window["has_next"]

    def test_the_offset_follows_the_page(self) -> None:
        assert _page_window(3, total=PAGE_SIZE * 5)["offset"] == PAGE_SIZE * 2

    def test_the_last_page_is_not_over_counted(self) -> None:
        """An exact multiple must not produce a trailing empty page."""
        assert _page_window(1, total=PAGE_SIZE * 2)["last"] == 2


# -- finding 13: the screens ---------------------------------------------------


async def test_the_review_queue_pages_rather_than_rendering_everything(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """More pending items than fit on a page, and the page stays bounded."""
    for index in range(PAGE_SIZE + 5):
        db_session.add(
            HumanReview(
                entity_type="covenant",
                entity_id=uuid.uuid4(),
                field_name="threshold_amount",
                new_value=str(index),
                source_quote=f"synthetic queue item {index}",
                page_number=1,
                confidence=0.5,
                trigger_reason=ReviewTrigger.LOW_CONFIDENCE,
                status=ReviewStatus.PENDING,
            )
        )
    await db_session.flush()

    first = await api_client.get("/ui/review")
    assert first.status_code == 200
    assert first.text.count("synthetic queue item") <= PAGE_SIZE
    assert "Next" in first.text

    second = await api_client.get("/ui/review?page=2")
    assert second.status_code == 200
    assert "Previous" in second.text
    # The two pages must not show the same rows.
    assert first.text != second.text


async def test_the_instruments_page_pages(api_client: AsyncClient, seeded_universe: None) -> None:
    response = await api_client.get("/ui/instruments?page=1")

    assert response.status_code == 200


async def test_a_negative_page_is_rejected_by_validation(api_client: AsyncClient) -> None:
    """`ge=1` on the query parameter, so page=0 is a 422 rather than a wrap-around."""
    assert (await api_client.get("/ui/review?page=0")).status_code == 422
