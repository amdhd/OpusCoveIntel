"""Catalogue read endpoints — instruments, covenants, provenance, portfolios.

The assertions that matter here are the invariants, not the plumbing:

* every covenant served carries its source page and verbatim quote
  (CLAUDE.md 1.2) -- a threshold without evidence must be unrenderable, and
* the provenance endpoint locates the quote inside the chunk, so a citation can
  be checked rather than merely displayed.

`indexed_corpus` ingests, indexes and rule-extracts all three synthetic
documents, so these run against real extracted rows rather than hand-built
fixtures -- which is the only way the clause join gets exercised honestly.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import Clause, Covenant
from app.db.models.instruments import Instrument
from app.db.models.portfolio import Portfolio

pytestmark = pytest.mark.usefixtures("storage_root")


async def _an_instrument(session: AsyncSession) -> Instrument:
    instrument = (await session.scalars(select(Instrument))).first()
    assert instrument is not None, "seeded_universe should provide instruments"
    return instrument


class TestInstruments:
    async def test_listing_returns_seeded_instruments(
        self, api_client: AsyncClient, seeded_universe: None
    ) -> None:
        response = await api_client.get("/instruments")
        assert response.status_code == 200
        body = response.json()
        assert body, "the seed creates instruments"
        assert {"issuer_name", "instrument_name", "current_rating"} <= set(body[0])

    async def test_the_ordinal_rank_is_exposed(
        self, api_client: AsyncClient, seeded_universe: None
    ) -> None:
        """Rating comparison is ordinal, not lexical (CLAUDE.md 6).

        A client given only `current_rating` would sort 'AA-' below 'A+'. The
        rank is what makes correct ordering possible on the other side.
        """
        body = (await api_client.get("/instruments")).json()
        rated = [row for row in body if row["current_rating"] is not None]
        assert rated, "the seed rates its instruments"
        assert all(row["current_rating_rank"] is not None for row in rated)

    async def test_the_canonical_notch_is_exposed(
        self, api_client: AsyncClient, seeded_universe: None
    ) -> None:
        """The rank orders ratings; the notch names what a rank *is*.

        A client rendering RAM's `AA3` beside MARC's `AA-` has to be able to say
        they are the same notch, and cannot without shipping the scale table --
        which would put a second copy of an ordering that breach evaluation
        depends on in TypeScript.
        """
        body = (await api_client.get("/instruments")).json()
        rated = [row for row in body if row["current_rating"] is not None]
        assert rated, "the seed rates its instruments"
        assert all(row["current_rating_notch"] is not None for row in rated)

        by_rating = {row["current_rating"]: row["current_rating_notch"] for row in rated}
        assert by_rating["AA3"] == "AA-", "RAM's numeric modifier maps onto the same notch"
        assert by_rating["A-"] == "A-", "a rating already on the canonical scale is unchanged"

    async def test_detail_assembles_the_related_records(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        instrument = await _an_instrument(db_session)
        response = await api_client.get(f"/instruments/{instrument.id}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == str(instrument.id)
        for key in ("covenants", "call_schedules", "rating_triggers"):
            assert key in body

    async def test_rating_triggers_come_back_tightest_first(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        instrument = await _an_instrument(db_session)
        triggers = (await api_client.get(f"/instruments/{instrument.id}")).json()["rating_triggers"]
        ranks = [trigger["trigger_rank"] for trigger in triggers]
        assert ranks == sorted(ranks)

    async def test_an_unknown_instrument_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.get(f"/instruments/{uuid.uuid4()}")
        assert response.status_code == 404


class TestCovenantsCarryTheirEvidence:
    async def test_every_covenant_names_its_page_and_quote(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        """CLAUDE.md 1.2 as an assertion, over real extracted rows."""
        covenant = (await db_session.scalars(select(Covenant))).first()
        assert covenant is not None, "the corpus should yield covenants"
        assert covenant.instrument_id is not None

        response = await api_client.get(f"/instruments/{covenant.instrument_id}/covenants")
        assert response.status_code == 200
        body = response.json()
        assert body, "this instrument has covenants"

        for row in body:
            source = row["source"]
            assert source["page_number"] >= 1
            assert source["source_quote"].strip()
            assert source["document_id"]
            assert source["clause_id"]

    async def test_filtering_by_type_narrows_the_result(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        covenant = (await db_session.scalars(select(Covenant))).first()
        assert covenant is not None
        wanted = covenant.covenant_type.value

        body = (
            await api_client.get(
                f"/instruments/{covenant.instrument_id}/covenants",
                params={"covenant_type": wanted},
            )
        ).json()
        assert body
        assert {row["covenant_type"] for row in body} == {wanted}

    async def test_a_monetary_threshold_survives_as_an_exact_decimal(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        """Money is Decimal, never float (CLAUDE.md 6), including on the wire."""
        covenant = (
            await db_session.scalars(select(Covenant).where(Covenant.threshold_amount.is_not(None)))
        ).first()
        assert covenant is not None, "the corpus contains a monetary threshold"

        body = (await api_client.get(f"/instruments/{covenant.instrument_id}/covenants")).json()
        served = [row for row in body if row["id"] == str(covenant.id)]
        assert served, "the covenant with a threshold should be served"
        amount = served[0]["threshold_amount"]

        # A JSON *string*, not a number. Pydantic does this for Decimal by
        # default; the assertion pins it rather than establishes it. Worth
        # pinning because a numeric literal would hand every browser a double
        # no matter what Postgres stored -- CLAUDE.md 6 undone at the boundary,
        # and invisible until a number is large enough to round.
        assert isinstance(amount, str), f"money must serialise as a string, got {type(amount)}"
        assert Decimal(amount) == covenant.threshold_amount

    async def test_an_unknown_instrument_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.get(f"/instruments/{uuid.uuid4()}/covenants")
        assert response.status_code == 404


class TestProvenance:
    async def test_a_clause_returns_its_chunk_and_locates_the_quote(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        clause = (
            await db_session.scalars(select(Clause).where(Clause.source_chunk_id.is_not(None)))
        ).first()
        assert clause is not None

        response = await api_client.get(f"/clauses/{clause.id}")
        assert response.status_code == 200
        body = response.json()

        assert body["chunk_text"], "provenance without the chunk cannot be checked"
        assert body["source"]["source_quote"]

        # The offsets must actually bracket the quote inside the chunk text --
        # an offset pair that points somewhere else would highlight the wrong
        # span for an auditor, which is worse than not highlighting at all.
        if body["quote_start"] is not None:
            start, end = body["quote_start"], body["quote_end"]
            assert body["chunk_text"][start:end] == body["source"]["source_quote"]

    async def test_provenance_includes_the_covenants_derived_from_the_clause(
        self, api_client: AsyncClient, db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
    ) -> None:
        covenant = (await db_session.scalars(select(Covenant))).first()
        assert covenant is not None

        body = (await api_client.get(f"/clauses/{covenant.clause_id}")).json()
        assert str(covenant.id) in {row["id"] for row in body["covenants"]}

    async def test_an_unknown_clause_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.get(f"/clauses/{uuid.uuid4()}")
        assert response.status_code == 404


class TestPortfolios:
    async def test_holdings_report_the_date_they_used(
        self, api_client: AsyncClient, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        portfolio = (await db_session.scalars(select(Portfolio))).first()
        assert portfolio is not None

        response = await api_client.get(f"/portfolios/{portfolio.id}/holdings")
        assert response.status_code == 200
        body = response.json()

        assert body["as_of_date"] is not None, "a caller cannot report exposure for an unknown date"
        assert body["count"] == len(body["holdings"])

    async def test_each_holding_resolves_its_instrument(
        self, api_client: AsyncClient, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        portfolio = (await db_session.scalars(select(Portfolio))).first()
        assert portfolio is not None

        body = (await api_client.get(f"/portfolios/{portfolio.id}/holdings")).json()
        assert body["holdings"], "the seed creates holdings"
        for holding in body["holdings"]:
            assert holding["instrument"]["instrument_name"]

    async def test_total_market_value_sums_the_priced_positions(
        self, api_client: AsyncClient, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        portfolio = (await db_session.scalars(select(Portfolio))).first()
        assert portfolio is not None

        body = (await api_client.get(f"/portfolios/{portfolio.id}/holdings")).json()
        priced = [
            Decimal(holding["market_value"])
            for holding in body["holdings"]
            if holding["market_value"] is not None
        ]
        assert priced, "the seed prices its holdings"
        assert Decimal(body["total_market_value"]) == sum(priced)

    async def test_an_unknown_portfolio_is_404(self, api_client: AsyncClient) -> None:
        response = await api_client.get(f"/portfolios/{uuid.uuid4()}/holdings")
        assert response.status_code == 404
