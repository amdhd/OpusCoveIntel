"""The request-scoped session must commit what a route wrote.

This exists because of a defect that every other test in the suite was blind
to. `get_session()` yielded a session and closed it without committing, so
`POST /review/{id}/approve` flushed its UPDATE, returned `200 {"status":
"approved"}`, and then discarded the write when the session closed. The review
queue looked like it worked and persisted nothing.

The API tests could not catch it: `api_client` overrides `get_session` with the
rolled-back `db_session` fixture, which is right for isolation and precisely
what hides a missing commit. So these tests drive the real dependency against
the real database and then look for the row from a *different* session, which
is the only vantage point from which "did it commit?" is a meaningful question.

They clean up after themselves rather than rolling back, because rolling back
is the bug.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db.models.ops import HumanReview
from app.domain.enums import ReviewStatus, ReviewTrigger

pytestmark = pytest.mark.usefixtures("_point_settings_at_test_db")


@pytest.fixture
def _point_settings_at_test_db(db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `get_session()` open against the test database, not the dev one.

    `get_session` reads `DATABASE_URL` through the settings cache, and the
    autouse `_clear_settings_cache` fixture drops that cache around every test,
    so setting the env var here is enough to redirect it.
    """
    from app.core.config import get_settings
    from app.db.session import get_engine, get_sessionmaker

    monkeypatch.setenv("DATABASE_URL", db_engine.url.render_as_string(hide_password=False))
    for cache in (get_settings, get_engine, get_sessionmaker):
        cache.cache_clear()


@pytest_asyncio.fixture
async def committed(db_engine: AsyncEngine) -> AsyncIterator[list[uuid.UUID]]:
    """Entity ids to delete on teardown.

    Every other test in the suite is isolated by rolling its transaction back.
    These cannot be -- a rollback is the very thing they exist to detect -- so
    they commit into the shared test database and must clean up after
    themselves instead.

    Cleanup lives in a fixture rather than a `finally` inside each test because
    an earlier draft put it in the test body, below the setup that commits. A
    failure in between leaked the row, and six leaked rows are enough to break
    every unrelated test that asserts an empty review queue. Teardown runs
    however the test exits; a `finally` two lines too low does not.

    Register the id before creating the row.
    """
    tracked: list[uuid.UUID] = []
    try:
        yield tracked
    finally:
        async with AsyncSession(db_engine) as cleanup:
            await cleanup.execute(delete(HumanReview).where(HumanReview.entity_id.in_(tracked)))
            await cleanup.commit()


async def _drain(session_dep: AsyncIterator[AsyncSession]) -> None:
    """Run a FastAPI yield-dependency to completion, as the framework would."""
    with pytest.raises(StopAsyncIteration):
        await session_dep.__anext__()


class TestGetSessionCommits:
    async def test_a_row_added_through_the_dependency_survives_the_request(
        self, db_engine: AsyncEngine, committed: list[uuid.UUID]
    ) -> None:
        """The whole point: another connection can see it afterwards."""
        from app.db.session import get_session

        entity_id = uuid.uuid4()
        committed.append(entity_id)

        dependency = get_session()
        session = await dependency.__anext__()
        session.add(
            HumanReview(
                entity_type="clause",
                entity_id=entity_id,
                field_name="threshold_amount",
                old_value="RM30,000,000",
                trigger_reason=ReviewTrigger.RULE_LLM_DISAGREEMENT,
                status=ReviewStatus.PENDING,
            )
        )
        await _drain(dependency)

        async with AsyncSession(db_engine) as observer:
            found = await observer.scalar(
                select(HumanReview).where(HumanReview.entity_id == entity_id)
            )
            assert found is not None, "the dependency closed without committing"
            assert found.status is ReviewStatus.PENDING

    async def test_an_update_through_the_dependency_survives_the_request(
        self, db_engine: AsyncEngine, committed: list[uuid.UUID]
    ) -> None:
        """The review-approval shape: read a row, mutate it, let the request end.

        This is the exact sequence `POST /review/{id}/approve` performs, minus
        the HTTP layer -- the route flushes and returns, and the dependency is
        the only thing left that could commit.
        """
        from app.db.session import get_session

        entity_id = uuid.uuid4()
        committed.append(entity_id)

        # expire_on_commit=False so reading `review.id` afterwards does not
        # trigger a lazy reload, which needs async I/O from a sync attribute.
        async with AsyncSession(db_engine, expire_on_commit=False) as setup:
            review = HumanReview(
                entity_type="covenant",
                entity_id=entity_id,
                field_name="covenant",
                old_value="1.75",
                trigger_reason=ReviewTrigger.LOW_CONFIDENCE,
                status=ReviewStatus.PENDING,
            )
            setup.add(review)
            await setup.commit()
            review_id = review.id

        dependency = get_session()
        session = await dependency.__anext__()
        subject = await session.get(HumanReview, review_id)
        assert subject is not None
        subject.status = ReviewStatus.APPROVED
        subject.reviewer_id = "regression-probe"
        # `resolved_review_has_reviewer` requires both, as the route sets both.
        subject.reviewed_at = dt.datetime.now(dt.UTC)
        await session.flush()
        await _drain(dependency)

        async with AsyncSession(db_engine) as observer:
            after = await observer.get(HumanReview, review_id)
            assert after is not None
            assert after.status is ReviewStatus.APPROVED, (
                "the approval was flushed but never committed"
            )
            assert after.reviewer_id == "regression-probe"

    async def test_a_failed_request_rolls_back(
        self, db_engine: AsyncEngine, committed: list[uuid.UUID]
    ) -> None:
        """Committing on the way out must not mean committing a failed request.

        A route that flushes and then raises -- `HTTPException` for a 409, say --
        must leave nothing behind. FastAPI throws the exception into the
        dependency, so the rollback belongs there.
        """
        from app.db.session import get_session

        entity_id = uuid.uuid4()
        # Registered even though the point is that nothing should persist: if
        # the rollback regresses, the leak is cleaned up rather than breaking
        # every later test that expects an empty queue.
        committed.append(entity_id)

        # `get_session` is annotated AsyncIterator but is an async generator;
        # `athrow` is how FastAPI delivers a handler's exception to it.
        dependency = cast("AsyncGenerator[AsyncSession, None]", get_session())
        session = await dependency.__anext__()
        session.add(
            HumanReview(
                entity_type="clause",
                entity_id=entity_id,
                field_name="operator",
                trigger_reason=ReviewTrigger.CITATION_UNVERIFIED,
                status=ReviewStatus.PENDING,
            )
        )
        await session.flush()

        boom = RuntimeError("route failed after flushing")
        with pytest.raises(RuntimeError):
            await dependency.athrow(boom)

        async with AsyncSession(db_engine) as observer:
            found = await observer.scalar(
                select(HumanReview).where(HumanReview.entity_id == entity_id)
            )
            assert found is None, "a failed request committed its partial write"
