"""The server-rendered UI.

These are not screenshot tests. They cover the handful of behaviours where the
UI could be wrong in a way that matters:

* an anonymous browser is redirected to the login form, not handed a 401 body,
* the `next` parameter cannot be turned into an open redirect,
* extracted text is escaped before it reaches the page,
* the provenance highlight brackets the real quote, and
* hiding the review buttons from an analyst is courtesy -- the server refuses
  the POST regardless.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.auth import User
from app.db.models.clauses import Clause
from app.db.models.ops import HumanReview
from app.domain.enums import ReviewStatus, ReviewTrigger, UserRole
from app.web.deps import safe_next

pytestmark = pytest.mark.usefixtures("storage_root")


def _analyst() -> User:
    return User(
        id=uuid.uuid4(),
        username="junior",
        display_name="Junior Analyst",
        password_hash="scrypt$test",
        role=UserRole.ANALYST,
        is_active=True,
    )


async def _pending(session: AsyncSession, **kwargs: object) -> HumanReview:
    review = HumanReview(
        entity_type="covenant",
        entity_id=uuid.uuid4(),
        field_name="threshold_amount",
        old_value="RM30,000,000",
        trigger_reason=ReviewTrigger.LOW_CONFIDENCE,
        status=ReviewStatus.PENDING,
        **kwargs,
    )
    session.add(review)
    await session.flush()
    return review


# -- redirects ---------------------------------------------------------------


class TestSafeNext:
    @pytest.mark.parametrize(
        "hostile",
        [
            "https://evil.example/steal",
            "//evil.example/steal",
            "http://evil.example",
            "javascript:alert(1)",
            "evil.example",
            "",
            None,
        ],
    )
    def test_a_non_local_next_is_discarded(self, hostile: str | None) -> None:
        """An open redirect on a login page is a phishing primitive.

        `//evil.example` is the case a naive startswith('/') check waves
        through: browsers treat a protocol-relative URL as absolute.
        """
        assert safe_next(hostile) == "/ui/ask"

    @pytest.mark.parametrize("local", ["/ui/review", "/ui/instruments?limit=5"])
    def test_a_local_path_survives(self, local: str) -> None:
        assert safe_next(local) == local


class TestAnonymousBrowsing:
    @pytest.mark.parametrize("path", ["/ui/ask", "/ui/instruments", "/ui/portfolios", "/ui/review"])
    async def test_a_page_redirects_to_login(
        self, anonymous_client: AsyncClient, path: str
    ) -> None:
        """A person following a bookmark cannot act on a JSON 401."""
        response = await anonymous_client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith("/ui/login")

    async def test_the_wanted_path_is_carried_into_the_login_link(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get("/ui/review", follow_redirects=False)
        assert "next=%2Fui%2Freview" in response.headers["location"]

    async def test_the_json_api_still_answers_401(self, anonymous_client: AsyncClient) -> None:
        """The redirect is for HTML pages only; API clients keep their status."""
        assert (await anonymous_client.get("/instruments")).status_code == 401


# -- rendering ---------------------------------------------------------------


class TestPagesRender:
    async def test_the_ask_page_renders(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/ui/ask")
        assert response.status_code == 200
        assert "<textarea" in response.text

    async def test_instruments_list_renders(
        self, api_client: AsyncClient, seeded_universe: None
    ) -> None:
        response = await api_client.get("/ui/instruments")
        assert response.status_code == 200
        assert "Green Ijarah" in response.text

    async def test_a_portfolio_reports_its_as_of_date(
        self, api_client: AsyncClient, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        from app.db.models.portfolio import Portfolio

        portfolio = (await db_session.scalars(select(Portfolio))).first()
        assert portfolio is not None

        response = await api_client.get(f"/ui/portfolios/{portfolio.id}")
        assert response.status_code == 200
        assert "As of" in response.text

    async def test_an_unknown_instrument_is_404(self, api_client: AsyncClient) -> None:
        assert (await api_client.get(f"/ui/instruments/{uuid.uuid4()}")).status_code == 404


class TestProvenancePage:
    async def test_the_highlight_wraps_the_quote(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        """`<mark>` must contain the quote, not merely appear on the page."""
        clause = (
            await db_session.scalars(select(Clause).where(Clause.source_chunk_id.is_not(None)))
        ).first()
        assert clause is not None

        response = await api_client.get(f"/ui/clauses/{clause.id}")
        assert response.status_code == 200

        body = response.text
        assert "<mark>" in body, "the quote should be highlighted within its chunk"
        marked = body.split("<mark>", 1)[1].split("</mark>", 1)[0]
        # Escaped in the page, so compare on a distinctive escaped fragment.
        head = clause.source_quote.strip()[:40].replace("&", "&amp;").replace("<", "&lt;")
        assert head in marked


class TestEscaping:
    async def test_extracted_text_is_escaped(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Clause text comes out of third-party PDFs; it is not trusted markup.

        A prospectus containing something script-shaped must render as text.
        """
        await _pending(db_session, source_quote="<script>alert('xss')</script> RM30,000,000")

        response = await api_client.get("/ui/review")
        assert response.status_code == 200
        assert "<script>alert(" not in response.text
        assert "&lt;script&gt;" in response.text


# -- review actions ----------------------------------------------------------


class TestReviewActions:
    async def test_approving_redirects_and_persists(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _pending(db_session)

        response = await api_client.post(f"/ui/review/{review.id}/approve", follow_redirects=False)
        # 303 to a GET so a refresh does not decide the item twice.
        assert response.status_code == 303
        assert response.headers["location"] == "/ui/review"

        await db_session.refresh(review)
        assert review.status is ReviewStatus.APPROVED
        assert review.reviewer_id == "test-reviewer"

    async def test_correcting_records_the_new_value(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _pending(db_session)

        response = await api_client.post(
            f"/ui/review/{review.id}/correct",
            data={"new_value": "RM50,000,000"},
            follow_redirects=False,
        )
        assert response.status_code == 303

        await db_session.refresh(review)
        assert review.status is ReviewStatus.CORRECTED
        assert review.old_value == "RM30,000,000", "the machine's value must survive"
        assert review.new_value == "RM50,000,000"

    async def test_deciding_twice_redirects_rather_than_erroring(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """A double submit is a 409 underneath; a person should just see the queue."""
        review = await _pending(db_session)
        await api_client.post(f"/ui/review/{review.id}/approve", follow_redirects=False)

        second = await api_client.post(f"/ui/review/{review.id}/approve", follow_redirects=False)
        assert second.status_code == 303
        assert second.headers["location"] == "/ui/review"

    async def test_an_unknown_action_is_404(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _pending(db_session)
        response = await api_client.post(f"/ui/review/{review.id}/delete")
        assert response.status_code == 404


@pytest.fixture
def as_analyst(api_app: Any) -> None:
    """Sign the UI in as an analyst for the duration of one test."""
    from app.web.deps import page_user

    api_app.dependency_overrides[page_user] = _analyst


@pytest.mark.usefixtures("as_analyst")
class TestAnalystCannotDecide:
    async def test_the_buttons_are_hidden(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _pending(db_session)
        response = await api_client.get("/ui/review")

        assert response.status_code == 200
        # Anchored on the banner's own words. A bare "read-only" also occurs in
        # the role chip's tooltip in every page's chrome, so it would pass with
        # the banner deleted -- which it did, once.
        assert "Read-only for your role" in response.text
        assert "Approve" not in response.text

    async def test_an_item_says_who_acts_next(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """An item with no buttons and no explanation reads as a broken card."""
        await _pending(db_session)
        response = await api_client.get("/ui/review")

        assert "Awaiting a reviewer's decision." in response.text

    async def test_the_queue_badge_does_not_call_an_analyst_to_action(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """The count is information for an analyst and a task for a reviewer."""
        await _pending(db_session)
        response = await api_client.get("/ui/review")

        assert 'class="pill passive"' in response.text
        assert "items awaiting a reviewer" in response.text

    async def test_the_server_refuses_the_post_anyway(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """A hidden form is not a permission."""
        review = await _pending(db_session)
        response = await api_client.post(f"/ui/review/{review.id}/approve")

        assert response.status_code == 403
        await db_session.refresh(review)
        assert review.status is ReviewStatus.PENDING


class TestReviewerChrome:
    """The same chrome, weighted for the role that can act on it."""

    async def test_the_queue_badge_is_a_call_to_action(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _pending(db_session)
        response = await api_client.get("/ui/review")

        assert 'class="pill "' in response.text, "no `passive` modifier for a reviewer"
        assert "items awaiting your decision" in response.text

    async def test_the_role_chip_says_what_the_role_may_do(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/ui/review")

        assert "may approve, correct and reject queue items" in response.text
