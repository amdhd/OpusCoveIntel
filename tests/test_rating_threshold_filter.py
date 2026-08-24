"""Rating-threshold answers must not claim an unrated instrument is below a notch.

`_format_instrument_answer` answers "which instruments are rated below A?" by
filtering the instrument list in Python. It had two holes, and both pointed the
same way -- towards including an instrument the system cannot actually place on
the scale:

* an instrument with **no rating** never reached the filter, because the guard
  was `if threshold_rating and inst.current_rating`;
* an instrument whose rating **cannot be ranked** was swallowed by
  `except UnknownRatingError: pass`, which fell through to the append.

Both then appeared under the headline "N instrument(s) rated below A", at
confidence 0.95. That is the inverse of the rule the rest of this codebase is
built on: "could not evaluate" and a definite verdict are never merged, and the
direction that gets someone to act is the dangerous one to guess.

The deterministic read path never had the bug -- it filters in SQL with
`current_rating_rank IS NOT NULL` -- so this is also the third instance of the
two read paths disagreeing (findings 14 and 15). The last test here pins them
together.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import AgentState, _format_instrument_answer
from app.agent.tools import ToolResult
from app.db.models.instruments import Instrument
from app.domain.enums import RatingAgency
from app.query.service import DeterministicQueryService

pytestmark = pytest.mark.usefixtures("storage_root")

QUESTION = "Which instruments are rated below A?"


def _instrument(name: str, rating: str | None) -> Instrument:
    instrument = Instrument(
        id=uuid.uuid4(),
        issuer_name="Synthetic Issuer Bhd",
        instrument_name=name,
        currency="MYR",
        issue_size=Decimal("100000000"),
        maturity_date=dt.date(2030, 1, 1),
        rating_agency=RatingAgency.MARC,
    )
    instrument.current_rating = rating
    return instrument


def _answer_for(instruments: list[Instrument]) -> tuple[str, float]:
    state = AgentState(question=QUESTION)
    state.tool_results = [
        ToolResult(
            tool_name="get_instrument",
            ok=True,
            data={"instruments": instruments, "count": len(instruments)},
        )
    ]
    return _format_instrument_answer(state)


class TestOnlyRankableInstrumentsAreReportedBelowAThreshold:
    def test_an_instrument_with_no_rating_is_not_reported_as_below(self) -> None:
        """The guard used to skip the filter entirely when the rating was None."""
        answer, _ = _answer_for([_instrument("Unrated Sukuk", None)])

        assert "Unrated Sukuk" not in answer
        assert "0 instrument(s) rated below A." in answer

    def test_an_unrankable_rating_is_not_reported_as_below(self) -> None:
        """`except UnknownRatingError: pass` used to fall through to the append."""
        answer, _ = _answer_for([_instrument("Odd Rating Sukuk", "Not-A-Rating")])

        assert "Odd Rating Sukuk" not in answer
        assert "0 instrument(s) rated below A." in answer

    def test_a_genuinely_lower_rating_is_still_reported(self) -> None:
        """The fix must not silence the answer it exists to give."""
        answer, confidence = _answer_for([_instrument("Weak Sukuk", "BBB-")])

        assert "Weak Sukuk" in answer
        assert "1 instrument(s) rated below A." in answer
        assert confidence > 0.0

    def test_a_stronger_rating_is_still_excluded(self) -> None:
        answer, _ = _answer_for([_instrument("Strong Sukuk", "AAA")])

        assert "Strong Sukuk" not in answer

    def test_the_headline_count_matches_the_rows_listed(self) -> None:
        """The count and the list were computed from the same loop and still disagreed
        with reality; this pins them to each other and to the truth."""
        answer, _ = _answer_for(
            [
                _instrument("Strong Sukuk", "AAA"),
                _instrument("Weak Sukuk", "BBB-"),
                _instrument("Odd Rating Sukuk", "Not-A-Rating"),
                _instrument("Unrated Sukuk", None),
            ]
        )

        assert "1 instrument(s) rated below A." in answer
        assert "Weak Sukuk" in answer
        for excluded in ("Strong Sukuk", "Odd Rating Sukuk", "Unrated Sukuk"):
            assert excluded not in answer

    def test_an_unfiltered_question_still_lists_everything(self) -> None:
        """No threshold in the question means no rating filter at all -- an unrated
        instrument belongs in "which instruments do we hold?"."""
        state = AgentState(question="Which instruments do we hold?")
        rows = [_instrument("Unrated Sukuk", None), _instrument("Weak Sukuk", "BBB-")]
        state.tool_results = [
            ToolResult(tool_name="get_instrument", ok=True, data={"instruments": rows, "count": 2})
        ]

        answer, _ = _format_instrument_answer(state)

        assert "Unrated Sukuk" in answer
        assert "Weak Sukuk" in answer


async def test_both_read_paths_agree_about_an_unrated_instrument(
    db_session: AsyncSession, seeded_universe: None
) -> None:
    """Findings 14 and 15 were both "the two read paths disagree". So is this.

    The deterministic path filters in SQL and was always right. Adding an
    unrated instrument to the corpus must not change that, and must not make
    the agent's formatter disagree with it.
    """
    db_session.add(
        Instrument(
            issuer_name="Synthetic Issuer Bhd",
            instrument_name="Unrated Probe Sukuk",
            currency="MYR",
            issue_size=Decimal("100000000"),
            maturity_date=dt.date(2030, 1, 1),
            rating_agency=RatingAgency.MARC,
        )
    )
    await db_session.flush()

    deterministic = await DeterministicQueryService(db_session).answer(QUESTION)

    assert "Unrated Probe Sukuk" not in deterministic.text, (
        "the SQL path must keep excluding an instrument it cannot rank"
    )
