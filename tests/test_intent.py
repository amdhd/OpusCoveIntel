"""Intent classification.

The class that matters is UNSUPPORTED. Everything else degrades gracefully --
a covenant question misrouted to document search still returns cited text --
but a forecast answered confidently is the failure mode that discredits the
whole system (PLAN.md 7).
"""

from __future__ import annotations

import pytest

from app.domain.enums import QueryIntent
from app.query.intent import classify, mentioned_entities


@pytest.mark.parametrize(
    "question",
    [
        "Should we buy more Malaysian sukuk next quarter?",
        "Should I sell the Wakalah sukuk?",
        "Will the sukuk market rally next year?",
        "What is your forecast for MGS yields?",
        "Can you predict the next rating action?",
        "Is this a good investment?",
        "What is the fair value of the RM300m Green Ijarah Sukuk?",
        "Do you recommend the Retail REIT sukuk?",
    ],
)
def test_advice_and_forecasts_are_unsupported(question: str) -> None:
    assert classify(question) is QueryIntent.UNSUPPORTED


def test_refusal_beats_a_perfectly_answerable_looking_question() -> None:
    # Contains "sukuk" and "rated", which would otherwise route to an
    # instrument lookup and produce a confident, useless answer.
    question = "Should we buy the sukuk rated A- by MARC?"

    assert classify(question) is QueryIntent.UNSUPPORTED


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which holdings breach their covenants?", QueryIntent.COVENANT_BREACH_CHECK),
        (
            "Is the issuer in compliance with its gearing covenant?",
            QueryIntent.COVENANT_BREACH_CHECK,
        ),
        ("How much headroom is there on the gearing covenant?", QueryIntent.COVENANT_BREACH_CHECK),
        ("What is the total exposure of the Green Fixed Income Fund?", QueryIntent.PORTFOLIO_QUERY),
        ("How much do we hold of the Wakalah sukuk?", QueryIntent.PORTFOLIO_QUERY),
        ("What is the cross-default threshold?", QueryIntent.COVENANT_LOOKUP),
        ("Apakah nisbah gearan yang perlu dikekalkan?", QueryIntent.COVENANT_LOOKUP),
        ("When can the issuer redeem the sukuk?", QueryIntent.COVENANT_LOOKUP),
        ("Which instruments are rated below A?", QueryIntent.INSTRUMENT_LOOKUP),
        ("What is the maturity of the Wakalah sukuk?", QueryIntent.INSTRUMENT_LOOKUP),
        ("Tell me about the trustee arrangements", QueryIntent.DOCUMENT_SEARCH),
    ],
)
def test_answerable_questions_route_to_the_right_intent(
    question: str, expected: QueryIntent
) -> None:
    assert classify(question) is expected


def test_an_empty_question_is_unsupported_rather_than_a_search() -> None:
    assert classify("   ") is QueryIntent.UNSUPPORTED


def test_a_breach_question_outranks_the_covenant_words_it_contains() -> None:
    # "gearing covenant" alone is a lookup; asking whether it is breached is a
    # rules-engine call, and the two produce very different answers.
    assert classify("Is the gearing covenant breached?") is QueryIntent.COVENANT_BREACH_CHECK


def test_entity_matching_is_literal() -> None:
    names = ["Synthetic Green Energy Sdn Bhd", "Synthetic Retail REIT Berhad"]

    found = mentioned_entities("What covenants apply to synthetic green energy sdn bhd?", names)

    assert found == ["Synthetic Green Energy Sdn Bhd"]


def test_an_unmentioned_entity_is_not_matched() -> None:
    # Attaching an answer to the wrong issuer produces a confident, wrong
    # portfolio number -- worse than returning nothing.
    assert mentioned_entities("What covenants apply?", ["Synthetic Retail REIT Berhad"]) == []
